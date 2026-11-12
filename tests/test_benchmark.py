import pandas as pd
import pytest

from streaming.benchmark import MEASURE_FILE, main, measured_rows, summarize

pytest.importorskip("matplotlib")


def _df():
    return pd.DataFrame({
        "flow_id": [1, 2, 3, 4], "orig_ts": "x", "label": ["Benign", "Bot", "Benign", "Bot"],
        "recon_error": [0.1, 5.0, 2.0, 0.2], "is_anomaly": [False, True, True, False],
        "produce_ts": [0, 500, 1500, 2000], "kafka_ts": [10, 510, 1510, 2010], "detect_ts": [1000, 1000, 2000, 3000]})


def test_summarize():
    s = summarize(_df())
    assert s["n_rows"] == 4 and s["duration_s"] == 2.0 and s["throughput_rps"] == 2.0
    assert s["lat_max_ms"] == 1000 and s["lat_p50_ms"] == pytest.approx(750)
    assert s["kafka_p50_ms"] == pytest.approx(740)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_measured_rows_uses_marker_file(tmp_path):
    df, since, dropped = measured_rows(_df(), tmp_path)
    assert since is None and dropped == 0 and len(df) == 4
    (tmp_path / MEASURE_FILE).write_text("1000\n", encoding="utf-8")
    df, since, dropped = measured_rows(_df(), tmp_path)
    assert since == 1000 and dropped == 2 and df["flow_id"].tolist() == [3, 4]
    df, since, dropped = measured_rows(_df(), tmp_path, since_ms=0)
    assert len(df) == 4


def test_cli_appends_and_plots(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _df().to_parquet(out / "batch-00000001.parquet")
    csv = tmp_path / "bench.csv"
    for rate, w in ((1000, 1), (1000, 2), (5000, 2)):
        main(["--out-dir", str(out), "--rate", str(rate), "--trigger", "2", "--n-workers", str(w), "--csv", str(csv)])
    df = pd.read_csv(csv)
    assert len(df) == 3 and {"rate", "n_workers", "throughput_rps", "lat_p95_ms", "f1", "warmup_rows_dropped"} <= set(df.columns)
    main(["--plot", "--csv", str(csv), "--reports-dir", str(tmp_path / "rep")])
    for f in ("stream_throughput.png", "stream_latency.png", "stream_workers.png"):
        assert (tmp_path / "rep" / f).exists()
