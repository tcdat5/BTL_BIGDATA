import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("onnx")
from common.fake import write_fake_dataset  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS, load_output  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder.master("local[2]").appName("test-stream") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()
    yield s
    s.stop()


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("sd")
    scaler = write_fake_dataset(d / "data", n=2000, scaler_path=d / "scaler.json")
    make_mean_square_onnx(d / "m.onnx", len(scaler.features))
    (d / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": 1.0}))
    return d, scaler


def _kafka_like_df(spark, d, scaler, n=30):
    df = pd.read_parquet(d / "data" / "test").head(n)
    rows = []
    for _, r in df.iterrows():
        msg = {"flow_id": int(r["flow_id"]), "produce_ts": 1000, "orig_ts": r["orig_ts"], "label": r["Label"],
               "features": {f: float(r[f]) for f in scaler.features}}
        rows.append((bytearray(json.dumps(msg).encode()), datetime(2026, 10, 14, 10, 0, 0)))
    return spark.createDataFrame(rows, "value binary, timestamp timestamp"), df


def test_parse_stream(spark, assets):
    from streaming.stream_detect import parse_stream
    d, scaler = assets
    raw, df = _kafka_like_df(spark, d, scaler)
    out = parse_stream(raw, scaler.features).toPandas()
    assert list(out.columns[:5]) == ["flow_id", "orig_ts", "label", "produce_ts", "kafka_ts"]
    assert out.shape[1] == 5 + len(scaler.features)
    np.testing.assert_allclose(out["f0"].to_numpy(), df[scaler.features[0]].to_numpy())
    assert (out["kafka_ts"] > 0).all()


@pytest.mark.parametrize("mode", ["driver", "executor"])
def test_batch_handler_writes_output(spark, assets, tmp_path, mode):
    from streaming.stream_detect import BatchHandler, parse_stream, prepare_executor_assets
    d, scaler = assets
    args = SimpleNamespace(model=str(d / "m.onnx"), scaler=str(d / "scaler.json"), threshold=str(d / "thr.json"),
                           out_dir=str(tmp_path / "out"), batches_csv=str(tmp_path / "b.csv"), inference=mode)
    (tmp_path / "out").mkdir()
    if mode == "executor":
        prepare_executor_assets(spark, args)
    raw, df = _kafka_like_df(spark, d, scaler)
    h = BatchHandler(args, len(scaler.features))
    h(parse_stream(raw, scaler.features), 7)
    out = load_output(tmp_path / "out")
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == len(df) and h.rows_seen == len(df)
    assert (tmp_path / "out" / "batch-00000007.parquet").exists()
    b = pd.read_csv(tmp_path / "b.csv")
    assert list(b.columns) == ["batch_id", "n_rows", "batch_start_ts", "batch_end_ts", "duration_ms"]
    assert b["n_rows"].tolist() == [len(df)]
