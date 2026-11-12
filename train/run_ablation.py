"""Ablation cho AE (PROJECT_SPEC 5.3). Mỗi model ablation train tối đa --epochs (mặc định 20,
thấp hơn 50 epoch của model chính) — ghi ở cột max_epochs.

python train/run_ablation.py --data-dir DATA --scaler models/scaler.json --only latent,threshold
Experiments: latent (4,8,16,32) | depth (32 / 64,32 / 128,64,32) | threshold (p95..p99.9, không train lại,
dùng score của AE chính trong --base-results-dir) | vae_beta (0.5,1,2).
Cấu hình mặc định của AE là hidden=64,32, latent=16: chỉ train một lần rồi dùng chung cho latent và depth.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from train import train_dl  # noqa: E402

ABLATION_COLUMNS = ["experiment", "value", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
                    "n_params", "train_time_s", "max_epochs"]
LATENTS = [4, 8, 16, 32]
DEPTHS = ["32", "64,32", "128,64,32"]
PERCENTILES = [95, 97, 99, 99.5, 99.9]
BETAS = [0.5, 1.0, 2.0]
EXPERIMENTS = ("latent", "depth", "threshold", "vae_beta")
_CACHE = {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--scaler", default=None)
    p.add_argument("--only", default="latent,depth,threshold", help="tập con của: " + ",".join(EXPERIMENTS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--base-max-epochs", type=int, default=50, help="max epoch của AE chính (cho dòng threshold)")
    p.add_argument("--test-cap", type=int, default=None)
    p.add_argument("--base-results-dir", default="results")
    p.add_argument("--out-root", default="results/ablation")
    p.add_argument("--out-csv", default="results/ablation.csv")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="dữ liệu giả, 2 epoch, CPU")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data_dir = a.data_dir or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    a.test_cap = a.test_cap or C.get(cfg, "data.test_cap", 300000)
    a.only = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in a.only if x not in EXPERIMENTS]
    if bad:
        p.error("experiment không hợp lệ: %s" % bad)
    return a


def train_variant(a, kind: str, hidden: str, latent: int, beta: float = 1.0):
    key = "%s_h%s_l%d_b%g" % (kind, hidden.replace(",", "-"), latent, beta)
    if key not in _CACHE:
        out = Path(a.out_root) / key
        argv = ["--model", kind, "--data-dir", str(a.data_dir), "--scaler", str(a.scaler),
                "--hidden", hidden, "--latent", str(latent), "--beta", str(beta),
                "--epochs", str(a.epochs), "--test-cap", str(a.test_cap),
                "--results-dir", str(out), "--models-dir", str(out), "--no-export"]
        if a.cpu:
            argv.append("--cpu")
        print("[ablation] train %s" % key, flush=True)
        _CACHE[key] = (train_dl.run(train_dl.parse_args(argv)), out)
    return _CACHE[key]


def to_row(experiment, value, m, n_params, train_time_s, max_epochs) -> dict:
    row = {"experiment": experiment, "value": value}
    row.update({k: m[k] for k in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")})
    row.update(n_params=int(n_params), train_time_s=float(train_time_s), max_epochs=int(max_epochs))
    return row


def threshold_rows(a) -> list:
    base = Path(a.base_results_dir)
    need = ["scores_DeepAutoencoder.npz", "valscores_DeepAutoencoder.npy", "dl_ae_metrics.csv"]
    max_ep = a.base_max_epochs
    if not all((base / f).exists() for f in need):
        print("[ablation] thiếu score AE chính trong %s -> train AE mặc định (%d epoch)" % (base, a.epochs))
        _, base = train_variant(a, "ae", "64,32", 16)
        max_ep = a.epochs
    d = np.load(base / "scores_DeepAutoencoder.npz")
    val = np.load(base / "valscores_DeepAutoencoder.npy")
    m0 = pd.read_csv(base / "dl_ae_metrics.csv").iloc[0]
    rows = []
    for p in PERCENTILES:
        ev = evaluate(d["y"], d["score"], float(np.percentile(val, p)))
        rows.append(to_row("threshold", p, ev, m0["n_params"], m0["train_time_s"], max_ep))
    return rows


def run_experiment(a, name: str) -> list:
    if name == "threshold":
        return threshold_rows(a)
    if name == "latent":
        grid = [("ae", "64,32", lat, 1.0, lat) for lat in LATENTS]
    elif name == "depth":
        grid = [("ae", h, 16, 1.0, h) for h in DEPTHS]
    else:
        grid = [("vae", "64,32", 16, b, b) for b in BETAS]
    rows = []
    for kind, hidden, latent, beta, value in grid:
        r, _ = train_variant(a, kind, hidden, latent, beta)
        rows.append(to_row(name, value, r, r["n_params"], r["train_time_s"], a.epochs))
    return rows


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    _CACHE.clear()
    if a.smoke:
        from common.fake import write_fake_dataset
        tmp = Path(tempfile.mkdtemp(prefix="ablation_smoke_"))
        write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
        a.data_dir, a.scaler, a.epochs, a.cpu = str(tmp / "data"), str(tmp / "scaler.json"), 2, True
    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=ABLATION_COLUMNS)
    keep = keep[~keep["experiment"].isin(a.only)]
    df = keep
    new_rows = []
    for name in a.only:
        print("[ablation] === %s" % name, flush=True)
        new_rows += run_experiment(a, name)
        frames = [f for f in (keep, pd.DataFrame(new_rows, columns=ABLATION_COLUMNS)) if len(f)]
        df = pd.concat(frames, ignore_index=True)[ABLATION_COLUMNS]
        df.to_csv(out, index=False)        # ghi sau mỗi experiment để không mất kết quả nếu Kaggle timeout
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
