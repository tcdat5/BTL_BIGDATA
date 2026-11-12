"""Đo throughput/latency/F1 từ output một lần chạy streaming (PROJECT_SPEC 5.7).

python3 streaming/benchmark.py --out-dir results/stream/r5000_w2 --rate 5000 --trigger 2 --n-workers 2
python3 streaming/benchmark.py --plot          # vẽ hình từ results/streaming_benchmark.csv
throughput = n / (max detect_ts - min detect_ts); latency e2e = detect_ts - produce_ts; Kafka->detect = detect_ts - kafka_ts.
Nếu <out-dir>/_MEASURE_START tồn tại (epoch ms, run_benchmark.sh ghi sau pha warm-up) thì chỉ tính các flow có
produce_ts >= mốc đó, để thời gian khởi động executor/ONNX không lẫn vào số đo (--since-ms ghi đè).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import binarize  # noqa: E402
from common.metrics import confusion_metrics  # noqa: E402
from streaming.output_io import load_output  # noqa: E402

MEASURE_FILE = "_MEASURE_START"


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        raise ValueError("output rỗng")
    span = (df["detect_ts"].max() - df["detect_ts"].min()) / 1000.0
    e2e = (df["detect_ts"] - df["produce_ts"]).to_numpy(dtype=np.float64)
    kd = (df["detect_ts"] - df["kafka_ts"]).to_numpy(dtype=np.float64)
    m = confusion_metrics(binarize(df["label"]), df["is_anomaly"])
    row = {"n_rows": n, "duration_s": span, "throughput_rps": n / span if span > 0 else float("nan")}
    for q in (50, 95, 99):
        row["lat_p%d_ms" % q] = float(np.percentile(e2e, q))
    row["lat_max_ms"] = float(e2e.max())
    for q in (50, 95, 99):
        row["kafka_p%d_ms" % q] = float(np.percentile(kd, q))
    row.update({k: m[k] for k in ("precision", "recall", "f1", "fpr")})
    return row


def measured_rows(df: pd.DataFrame, out_dir, since_ms=None):
    """Bỏ flow warm-up: giữ produce_ts >= since_ms (hoặc mốc trong <out_dir>/_MEASURE_START)."""
    if since_ms is None:
        f = Path(out_dir) / MEASURE_FILE
        if f.exists():
            since_ms = int(f.read_text(encoding="utf-8").strip())
    if since_ms is None:
        return df, None, 0
    keep = df["produce_ts"] >= since_ms
    return df[keep], since_ms, int((~keep).sum())


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists():
        new = pd.concat([pd.read_csv(csv_path), new], ignore_index=True, sort=False)
    new.to_csv(csv_path, index=False)


def plot(csv_path: Path, rep: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if "timestamp" in df:      # chạy lại cùng (rate, worker) thì lấy lần mới nhất
        df = df.sort_values("timestamp").groupby(["rate", "n_workers"]).tail(1)
    df = df.sort_values("rate")
    xcol = "actual_rate" if "actual_rate" in df and df["actual_rate"].notna().all() else "rate"

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for w, g in df.groupby("n_workers"):
        ax.plot(g[xcol], g["throughput_rps"], marker="o", label="%d worker" % w)
    lim = df[xcol].max()
    ax.plot([0, lim], [0, lim], ls=":", color="grey", label="lý tưởng (= rate gửi)")
    ax.set(xlabel="rate producer thực tế (msg/s)", ylabel="throughput (flow/s)", title="Throughput theo rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_throughput.png", dpi=150)
    plt.close(fig)

    wmax = df["n_workers"].max()
    g = df[df["n_workers"] == wmax]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for q in ("lat_p50_ms", "lat_p95_ms", "lat_p99_ms"):
        ax.plot(g[xcol], g[q], marker="o", label=q.replace("lat_", "").replace("_ms", ""))
    ax.set(xlabel="rate producer thực tế (msg/s)", ylabel="latency end-to-end (ms)", yscale="log",
           title="Latency theo rate (%d worker)" % wmax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_latency.png", dpi=150)
    plt.close(fig)

    def xlabel(rate):     # producer không đạt rate mục tiêu -> ghi kèm rate thực tế
        if "actual_rate" not in df:
            return "%d" % rate
        act = df.loc[df["rate"] == rate, "actual_rate"].mean()
        return "%d" % rate if not act == act or act >= 0.9 * rate else "%d\n(thực tế ~%d)" % (rate, act)
    piv_t = df.pivot_table(index="rate", columns="n_workers", values="throughput_rps")
    piv_l = df.pivot_table(index="rate", columns="n_workers", values="lat_p95_ms")
    labels = [xlabel(r) for r in piv_t.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    piv_t.plot.bar(ax=a1, rot=0)
    a1.set(xlabel="rate mục tiêu (msg/s)", ylabel="throughput (flow/s)", title="Throughput: số worker")
    piv_l.plot.bar(ax=a2, rot=0)
    a2.set(xlabel="rate mục tiêu (msg/s)", ylabel="latency p95 (ms)", title="Latency p95: số worker")
    a1.set_xticklabels(labels, fontsize=8)
    a2.set_xticklabels(labels, fontsize=8)
    for ax in (a1, a2):
        ax.legend(title="worker", fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_workers.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", help="thư mục output của một lần chạy stream_detect")
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--trigger", type=float, default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--max-offsets", type=int, default=None)
    p.add_argument("--inference", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--since-ms", type=int, default=None, help="chỉ tính flow có produce_ts >= mốc này")
    p.add_argument("--producer-stats", default=None, help="file JSON từ producer --stats-out")
    p.add_argument("--csv", default="results/streaming_benchmark.csv")
    p.add_argument("--no-append", action="store_true", help="chỉ in kết quả")
    p.add_argument("--plot", action="store_true", help="vẽ hình từ --csv rồi thoát")
    p.add_argument("--reports-dir", default="reports")
    a = p.parse_args(argv)
    if a.plot:
        plot(Path(a.csv), Path(a.reports_dir))
        return None
    if not a.out_dir:
        p.error("--out-dir là bắt buộc (trừ khi --plot)")
    df, since, n_warm = measured_rows(load_output(a.out_dir), a.out_dir, a.since_ms)
    s = summarize(df)
    actual = None
    if a.producer_stats and Path(a.producer_stats).exists():
        actual = json.loads(Path(a.producer_stats).read_text(encoding="utf-8")).get("actual_rate")
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": a.tag, "rate": a.rate, "actual_rate": actual,
           "trigger_s": a.trigger, "n_workers": a.n_workers, "max_offsets": a.max_offsets,
           "inference": a.inference, "warmup_rows_dropped": n_warm, **s}
    print(json.dumps(row, indent=1))
    if not a.no_append:
        append_row(Path(a.csv), row)
    return row


if __name__ == "__main__":
    main()
