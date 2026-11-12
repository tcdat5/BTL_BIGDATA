"""Replay Parquet test -> Kafka, mỗi flow một message JSON (PROJECT_SPEC 5.5).

python3 streaming/producer.py --rate 2000 --limit 100000
python3 streaming/producer.py --input hdfs:///.../processed/test --rate 0 --loop
python3 streaming/producer.py --input data/demo_test.parquet --rate 500
Gửi feature GỐC (chưa chuẩn hoá); stream_detect chuẩn hoá bằng scaler.json. 'label' chỉ dùng để đánh giá.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import ID_COL, LABEL_COL, TS_COL, load_scaler  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--bootstrap", default=None, help="mặc định cluster.kafka_bootstrap")
    p.add_argument("--topic", default=None, help="mặc định kafka.topic")
    p.add_argument("--input", default=None,
                   help="Parquet test (thư mục/file cục bộ hoặc hdfs://); mặc định <data.local_dir>/test")
    p.add_argument("--scaler", default=None, help="danh sách feature gửi đi; mặc định models.scaler")
    p.add_argument("--rate", type=float, default=1000.0, help="message/giây; 0 = không giới hạn")
    p.add_argument("--limit", type=int, default=0, help="số flow tối đa mỗi vòng; 0 = tất cả")
    p.add_argument("--loop", action="store_true", help="phát lại liên tục (flow_id cộng offset mỗi vòng)")
    p.add_argument("--create-topic", action="store_true")
    p.add_argument("--partitions", type=int, default=None)
    p.add_argument("--linger-ms", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=262144)
    p.add_argument("--report-every", type=float, default=5.0)
    p.add_argument("--stats-out", default=None)
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.bootstrap = a.bootstrap or C.get(cfg, "cluster.kafka_bootstrap")
    a.topic = a.topic or C.get(cfg, "kafka.topic")
    a.input = a.input or str(Path(C.get(cfg, "data.local_dir")) / "test")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    a.partitions = a.partitions or C.get(cfg, "kafka.partitions", 3)
    return a


class RateLimiter:
    """Message thứ n được gửi không sớm hơn t0 + n/rate."""

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.clock, self.sleep, self.t0 = rate, clock, sleep, None

    def wait(self, n_sent: int) -> None:
        if self.rate <= 0:
            return
        if self.t0 is None:
            self.t0 = self.clock()
        delay = self.t0 + n_sent / self.rate - self.clock()
        if delay > 0:
            self.sleep(delay)


def fetch_hdfs(path: str) -> Path:
    dest = ROOT / "data" / "cache" / path.rstrip("/").split("/")[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hdfs", "dfs", "-get", path, str(dest)], check=True)
    return dest


def load_frame(path, features, limit: int = 0) -> pd.DataFrame:
    local = fetch_hdfs(str(path)) if str(path).startswith("hdfs://") else Path(path)
    df = pd.read_parquet(local, columns=[ID_COL, TS_COL, LABEL_COL] + list(features))
    df = df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    return df.head(limit) if limit else df


def iter_messages(df: pd.DataFrame, features, id_offset: int = 0, chunk: int = 5000):
    """Đổi sang object Python theo từng khối: cả 480k dòng một lúc tốn > 1 GB RAM (master chỉ có 3.8 GB)."""
    features = list(features)
    for s in range(0, len(df), chunk):
        part = df.iloc[s:s + chunk]
        vals = part[features].to_numpy(dtype=np.float64).tolist()
        ids = part[ID_COL].astype("int64").tolist()
        ts = part[TS_COL].astype(str).tolist()
        labels = part[LABEL_COL].astype(str).tolist()
        for i in range(len(ids)):
            yield {"flow_id": ids[i] + id_offset, "orig_ts": ts[i], "label": labels[i],
                   "features": dict(zip(features, vals[i]))}


def ensure_topic(bootstrap: str, topic: str, partitions: int) -> None:
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError
    admin = KafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def run(a) -> dict:
    from kafka import KafkaProducer
    features = load_scaler(a.scaler).features
    df = load_frame(a.input, features, a.limit)
    if a.create_topic:
        ensure_topic(a.bootstrap, a.topic, a.partitions)
    prod = KafkaProducer(bootstrap_servers=a.bootstrap, linger_ms=a.linger_ms, batch_size=a.batch_size, acks=1,
                         value_serializer=lambda m: json.dumps(m, separators=(",", ":")).encode("utf-8"))
    lim = RateLimiter(a.rate)
    span = int(df[ID_COL].max()) + 1 if len(df) else 0
    sent, loop = 0, 0
    t0 = last_t = time.time()
    last_n = 0
    print("[producer] %d flow -> %s/%s, rate=%s" % (len(df), a.bootstrap, a.topic, a.rate or "max"), flush=True)
    try:
        while True:
            for msg in iter_messages(df, features, id_offset=loop * span):
                if sent % 50 == 0:
                    lim.wait(sent)
                msg["produce_ts"] = int(time.time() * 1000)
                prod.send(a.topic, value=msg)
                sent += 1
                now = time.time()
                if now - last_t >= a.report_every:
                    print("[producer] sent %d, rate %.0f msg/s" % (sent, (sent - last_n) / (now - last_t)), flush=True)
                    last_t, last_n = now, sent
            loop += 1
            if not a.loop:
                break
    except KeyboardInterrupt:
        print("[producer] dừng theo yêu cầu", flush=True)
    finally:
        prod.flush()
        prod.close()
    dur = time.time() - t0
    stats = {"sent": sent, "duration_s": round(dur, 3), "actual_rate": sent / dur if dur > 0 else 0.0,
             "target_rate": a.rate, "topic": a.topic}
    print("[producer] done: %s" % json.dumps(stats), flush=True)
    if a.stats_out:
        Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.stats_out).write_text(json.dumps(stats), encoding="utf-8")
    return stats


def main(argv=None) -> dict:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
