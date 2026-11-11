import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from common.data import load_scaler, transform  # noqa: E402
from common.fake import write_fake_dataset  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS, load_output  # noqa: E402
from streaming.scorer import PASSTHROUGH_COLS, OnnxScorer, get_scorer, score_frame  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402


@pytest.fixture()
def assets(tmp_path):
    scaler = write_fake_dataset(tmp_path / "data", n=1000, scaler_path=tmp_path / "scaler.json")
    make_mean_square_onnx(tmp_path / "m.onnx", len(scaler.features))
    (tmp_path / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": 1.0}))
    return tmp_path, scaler


def test_score_matrix_matches_numpy(assets):
    d, scaler = assets
    s = OnnxScorer(d / "m.onnx", d / "scaler.json", d / "thr.json")
    raw = pd.read_parquet(d / "data" / "test")[scaler.features].to_numpy(np.float64)
    expected = (transform(raw, load_scaler(d / "scaler.json")).astype(np.float64) ** 2).mean(axis=1)
    np.testing.assert_allclose(s.score_matrix(raw), expected, rtol=1e-5)
    assert s.threshold == 1.0


def test_get_scorer_is_cached(assets):
    d, _ = assets
    a = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    b = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    assert a is b


def test_score_frame_columns_and_flags(assets):
    d, scaler = assets
    s = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    df = pd.read_parquet(d / "data" / "test").head(50)
    pdf = pd.DataFrame({"flow_id": df["flow_id"], "orig_ts": df["orig_ts"], "label": df["Label"],
                        "produce_ts": 1, "kafka_ts": 2})
    for i, f in enumerate(scaler.features):
        pdf["f%d" % i] = df[f].to_numpy()
    out = score_frame(pdf, s, ["f%d" % i for i in range(len(scaler.features))])
    assert list(out.columns) == PASSTHROUGH_COLS + ["recon_error", "is_anomaly"]
    assert (out["is_anomaly"] == (out["recon_error"] >= 1.0)).all()
    empty = score_frame(pdf.head(0), s, ["f%d" % i for i in range(len(scaler.features))])
    assert len(empty) == 0 and list(empty.columns) == list(out.columns)


def test_load_output_ignores_tmp(tmp_path):
    row = {c: [1] for c in OUTPUT_COLUMNS}
    pd.DataFrame(row).to_parquet(tmp_path / "batch-00000001.parquet")
    (tmp_path / "batch-00000002.parquet.tmp").write_bytes(b"partial")
    assert len(load_output(tmp_path)) == 1
    assert list(load_output(tmp_path / "missing").columns) == OUTPUT_COLUMNS
