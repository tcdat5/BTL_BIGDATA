import json
import os
import uuid

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("onnx")
from common.data import load_split  # noqa: E402
from common.fake import write_fake_dataset  # noqa: E402
from common.metrics import confusion_metrics  # noqa: E402
from streaming import producer, stream_detect  # noqa: E402
from streaming.output_io import load_output  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402

pytestmark = [pytest.mark.spark, pytest.mark.kafka]
BOOT = os.environ.get("KAFKA_BOOTSTRAP")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not BOOT:
        pytest.skip("KAFKA_BOOTSTRAP chưa đặt")
    d = tmp_path_factory.mktemp("e2e")
    scaler = write_fake_dataset(d / "data", n=5000, scaler_path=d / "scaler.json")
    make_mean_square_onnx(d / "m.onnx", len(scaler.features))
    val = load_split(d / "data", "val", scaler, benign_only=True)
    thr = float(np.percentile((val.X.astype(np.float64) ** 2).mean(axis=1), 99))
    (d / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": thr}))
    return d, scaler, thr


@pytest.mark.parametrize("mode", ["driver", "executor"])
def test_stream_matches_offline(prepared, mode):
    d, scaler, thr = prepared
    topic = "e2e-%s-%s" % (mode, uuid.uuid4().hex[:8])
    producer.main(["--bootstrap", BOOT, "--topic", topic, "--create-topic", "--input", str(d / "data" / "test"),
                   "--scaler", str(d / "scaler.json"), "--rate", "0"])
    out = d / ("out_" + mode)
    res = stream_detect.main(["--master", "local[2]", "--bootstrap", BOOT, "--topic", topic,
                              "--model", str(d / "m.onnx"), "--threshold", str(d / "thr.json"),
                              "--scaler", str(d / "scaler.json"), "--out-dir", str(out),
                              "--checkpoint", str(d / ("ckpt_" + mode)), "--batches-csv", str(d / ("b_%s.csv" % mode)),
                              "--starting-offsets", "earliest", "--trigger", "1", "--max-offsets-per-trigger", "300",
                              "--inference", mode, "--stop-when-idle", "10", "--timeout", "300", "--clean"])
    te = load_split(d / "data", "test", scaler)
    exp = pd.DataFrame({"flow_id": te.meta["flow_id"], "err": (te.X.astype(np.float64) ** 2).mean(axis=1),
                        "y": te.y}).sort_values("flow_id").reset_index(drop=True)
    got = load_output(out).sort_values("flow_id").reset_index(drop=True)
    assert res["rows"] == len(exp) == len(got)
    np.testing.assert_array_equal(got["flow_id"].to_numpy(), exp["flow_id"].to_numpy())
    np.testing.assert_allclose(got["recon_error"].to_numpy(), exp["err"].to_numpy(), rtol=1e-5)
    f1_stream = confusion_metrics((got["label"] != "Benign").astype(int), got["is_anomaly"])["f1"]
    f1_off = confusion_metrics(exp["y"], exp["err"] >= thr)["f1"]
    assert abs(f1_stream - f1_off) < 0.01
    assert (got["detect_ts"] >= got["produce_ts"]).all()
    assert len(pd.read_csv(d / ("b_%s.csv" % mode))) >= 2     # nhiều micro-batch (maxOffsetsPerTrigger=300)
