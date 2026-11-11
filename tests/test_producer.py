import json
import os
import uuid

import pytest

from common.data import load_scaler
from common.fake import FAKE_FEATURES, write_fake_dataset
from streaming.producer import RateLimiter, iter_messages, load_frame, main


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def test_rate_limiter_paces():
    c = FakeClock()
    lim = RateLimiter(100, clock=c, sleep=c.sleep)
    lim.wait(0)
    lim.wait(50)
    assert c.t == pytest.approx(0.5)
    lim.wait(10)                      # đang trễ hơn lịch -> không ngủ
    assert c.t == pytest.approx(0.5)


def test_rate_limiter_unlimited():
    c = FakeClock()
    lim = RateLimiter(0, clock=c, sleep=c.sleep)
    lim.wait(10 ** 6)
    assert c.t == 0.0


def test_load_frame_and_messages(tmp_path):
    write_fake_dataset(tmp_path, n=2000, scaler_path=tmp_path / "scaler.json")
    feats = load_scaler(tmp_path / "scaler.json").features
    df = load_frame(tmp_path / "test", feats, limit=5)
    assert len(df) == 5 and df["flow_id"].is_monotonic_increasing
    msgs = list(iter_messages(df, feats, id_offset=1000))
    m = msgs[0]
    assert set(m) == {"flow_id", "orig_ts", "label", "features"}
    assert m["flow_id"] == int(df["flow_id"].iloc[0]) + 1000
    assert list(m["features"]) == FAKE_FEATURES
    assert m["features"][feats[0]] == float(df[feats[0]].iloc[0])     # feature gốc, chưa chuẩn hoá
    json.dumps(m)                     # serialize được


def test_load_frame_single_file(tmp_path):
    write_fake_dataset(tmp_path, n=500, scaler_path=tmp_path / "scaler.json")
    feats = load_scaler(tmp_path / "scaler.json").features
    assert len(load_frame(tmp_path / "test" / "part-00000.parquet", feats)) > 0   # vd data/demo_test.parquet


@pytest.mark.kafka
def test_produce_to_kafka(tmp_path):
    boot = os.environ.get("KAFKA_BOOTSTRAP")
    if not boot:
        pytest.skip("KAFKA_BOOTSTRAP chưa đặt")
    from kafka import KafkaConsumer
    write_fake_dataset(tmp_path, n=2000, scaler_path=tmp_path / "scaler.json")
    topic = "test-producer-" + uuid.uuid4().hex[:8]
    stats = main(["--bootstrap", boot, "--topic", topic, "--create-topic", "--input", str(tmp_path / "test"),
                  "--scaler", str(tmp_path / "scaler.json"), "--rate", "0", "--limit", "200",
                  "--stats-out", str(tmp_path / "p.json")])
    assert stats["sent"] == 200 and json.loads((tmp_path / "p.json").read_text())["sent"] == 200
    c = KafkaConsumer(topic, bootstrap_servers=boot, auto_offset_reset="earliest", consumer_timeout_ms=10000,
                      value_deserializer=lambda b: json.loads(b))
    got = [r.value for r in c]
    c.close()
    assert len(got) == 200 and all(isinstance(g["produce_ts"], int) for g in got)
