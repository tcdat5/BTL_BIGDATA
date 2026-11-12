"""Trực quan hoá dữ liệu CSE-CIC-IDS2018 (PROJECT_SPEC 5.10) -> <out>/*.png + eda_summary.md.

Phần 1 (sau preprocessing):
  python analysis/eda.py --data-dir data/kaggle_sample --scaler models/scaler.json --stats results/preprocess_stats.json
Phần 2 (sau train DL, cần models/ae.pt, models/vae.pt):
  python analysis/eda.py --latent --no-part1 --models-dir models
Smoke (dữ liệu giả):  python analysis/eda.py --smoke [--latent]
Feature có thể âm nên "thang log" là sign(x)*log1p(|x|).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.style import DPI, apply_style, label_colors  # noqa: E402
from common import config as C  # noqa: E402
from common.data import (BENIGN, CLIP, LABEL_COL, SEED, binarize, frame_to_matrix,  # noqa: E402
                         load_scaler, read_split_frame)

CORR_THRESHOLD = 0.95
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
DESCRIPTIONS = {
    "label_distribution.png": "Số dòng theo từng loại nhãn (trục log), tách train/val/test, đếm trên toàn bộ dữ liệu sau tiền xử lý.",
    "top_features_box.png": "Phân phối (thang sign·log1p) của các feature có KS Benign-vs-Attack lớn nhất, theo từng loại nhãn.",
    "correlation_heatmap.png": "Tương quan Pearson giữa các feature trên Benign train, sắp theo hierarchical clustering.",
    "scaling_before_after.png": "Histogram của 4 feature phân biệt nhất trước (sign·log1p) và sau chuẩn hoá + clip ±10.",
    "clip_rate.png": "Tỷ lệ giá trị bị clip ở ±10 sau chuẩn hoá, 20 feature cao nhất.",
    "pca_explained_variance.png": "Explained variance tích luỹ của PCA trên dữ liệu đã chuẩn hoá.",
    "projection_pca.png": "Chiếu PCA 2D của mẫu phân tầng, tô màu theo loại nhãn.",
    "projection_tsne.png": "Chiếu t-SNE 2D của cùng mẫu, tô màu theo loại nhãn.",
    "projection_umap.png": "Chiếu UMAP 2D của cùng mẫu, tô màu theo loại nhãn.",
    "latent_pca.png": "PCA 2D của dữ liệu gốc, latent AE và latent VAE (μ) trên cùng mẫu.",
    "latent_tsne.png": "t-SNE 2D của dữ liệu gốc, latent AE và latent VAE (μ) trên cùng mẫu.",
}


def log(msg):
    print("[eda] " + msg, flush=True)


def symlog1p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log("wrote " + Path(path).name)


def short_label(name: str) -> str:
    """'DDOS attack-HOIC' -> 'DDoS-HOIC', 'DoS attacks-SlowHTTPTest' -> 'DoS-SlowHTTPTest' (nhãn trục ngắn, không cắt cụt)."""
    return name.replace("DDOS attack-", "DDoS-").replace("DoS attacks-", "DoS-")


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), per_class), random_state=seed) for _, g in df.groupby(LABEL_COL, sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def scatter_by_label(ax, P, labels, title):
    colors = label_colors(labels, BENIGN)
    for lab in [BENIGN] + sorted(set(labels) - {BENIGN}):      # Benign vẽ trước (nằm dưới)
        m = labels == lab
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.6, color=colors[lab], label=lab, rasterized=True, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def tsne_2d(X, perplexity):
    from sklearn.manifold import TSNE
    perp = min(perplexity, max(5.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED).fit_transform(X)


# ---------- Phần 1 ----------
def label_distribution(stats, out: Path):
    ld = stats.get("label_distribution") if stats else None
    if not ld:
        return log("[skip] label distribution: thiếu preprocess_stats.json")
    df = pd.DataFrame(ld).fillna(0).astype(int)
    df = df.loc[df["train"].sort_values(ascending=True).index]
    df.rename_axis("label").to_csv(out / "label_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.5))
    y = np.arange(len(df))
    for i, split in enumerate(["train", "val", "test"]):
        if split in df:
            ax.barh(y + (i - 1) * 0.27, df[split].clip(lower=0.8), height=0.25, label=split, color=SPLIT_COLORS[split])
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xscale("log")
    ax.set(xlabel="số dòng (log)", title="Phân phối nhãn sau tiền xử lý")
    ax.grid(axis="y", visible=False)
    ax.legend()
    save(fig, out / "label_distribution.png")


def cleaning_steps(stats, out: Path):
    steps = stats.get("cleaning") if stats else None
    if not steps:
        return log("[skip] cleaning steps: preprocess_stats.json thiếu khoá 'cleaning'")
    pd.DataFrame(steps).to_csv(out / "cleaning_steps.csv", index=False)
    log("wrote cleaning_steps.csv")


def ks_analysis(train: pd.DataFrame, features, top_k: int, sample: pd.DataFrame, out: Path) -> list:
    from scipy.stats import ks_2samp
    attack = train[LABEL_COL].to_numpy() != BENIGN
    rows = []
    for f in features:
        x = train[f].to_numpy(dtype=np.float64)
        r = ks_2samp(x[~attack], x[attack]) if attack.any() and (~attack).any() else None
        rows.append({"feature": f, "ks": float(r.statistic) if r else np.nan, "pvalue": float(r.pvalue) if r else np.nan})
    ks = pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)
    ks.to_csv(out / "ks_table.csv", index=False)
    top = ks["feature"].head(top_k).tolist()
    groups = [BENIGN] + sorted(set(sample[LABEL_COL]) - {BENIGN})
    colors = label_colors(groups, BENIGN)
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, f in zip(axes.flat, top):
        data = [symlog1p(sample.loc[sample[LABEL_COL] == g, f]) for g in groups]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops={"color": "#111827", "linewidth": 1.2})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(colors[g])
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([short_label(g) for g in groups], rotation=45, ha="right", fontsize=7)
        ax.set_title("%s (KS=%.2f)" % (f, ks.set_index("feature").loc[f, "ks"]), fontsize=9)
        ax.set_ylabel("sign·log1p", fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(top):]:
        ax.axis("off")
    save(fig, out / "top_features_box.png")
    return top


def correlation(train: pd.DataFrame, features, out: Path) -> int:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    benign = train.loc[train[LABEL_COL] == BENIGN, list(features)]
    corr = benign.corr().fillna(0.0)
    cv = corr.to_numpy(copy=True)          # pandas >= 3 (copy-on-write) trả mảng chỉ đọc
    np.fill_diagonal(cv, 1.0)
    corr = pd.DataFrame(cv, index=corr.index, columns=corr.columns)
    dist = 1.0 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(np.clip(dist, 0, None), checks=False), method="average"))
    c = corr.iloc[order, order]
    n = len(c)
    fig, ax = plt.subplots(figsize=(0.16 * n + 4, 0.16 * n + 3))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fs = 8 if n <= 30 else 5
    ax.set_xticks(range(n))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(c.index, fontsize=fs)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Tương quan feature (Benign train)")
    save(fig, out / "correlation_heatmap.png")
    iu = np.triu_indices(n, k=1)
    vals = corr.to_numpy()[iu]
    mask = np.abs(vals) > CORR_THRESHOLD
    pairs = pd.DataFrame({"feature_a": corr.index[iu[0][mask]], "feature_b": corr.columns[iu[1][mask]], "r": vals[mask]})
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
    pairs.to_csv(out / "high_corr_pairs.csv", index=False)
    return len(pairs)


def scaling(train: pd.DataFrame, scaler, top: list, out: Path):
    Z = frame_to_matrix(train, scaler)
    rate = pd.DataFrame({"feature": scaler.features, "clip_rate": (np.abs(Z) >= CLIP).mean(axis=0)}) \
        .sort_values("clip_rate", ascending=False)
    rate.to_csv(out / "clip_rate.csv", index=False)
    head = rate.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(head) + 1.5))
    ax.barh(head["feature"], head["clip_rate"], color="#2a78d6", height=0.7)
    ax.set(xlabel="tỷ lệ giá trị bị clip ở ±%g" % CLIP, title="Clip sau chuẩn hoá (20 feature cao nhất)")
    ax.grid(axis="y", visible=False)
    save(fig, out / "clip_rate.png")
    feats = top[:4]
    idx = [scaler.features.index(f) for f in feats]
    fig, axes = plt.subplots(2, len(feats), figsize=(4 * len(feats), 6), squeeze=False)
    for j, (f, k) in enumerate(zip(feats, idx)):
        axes[0, j].hist(symlog1p(train[f]), bins=60, color="#2a78d6")
        axes[0, j].set_title(f, fontsize=9)
        axes[0, j].set_xlabel("gốc (sign·log1p)", fontsize=8)
        axes[1, j].hist(Z[:, k], bins=60, color="#2a78d6")
        for v in (-CLIP, CLIP):
            axes[1, j].axvline(v, color="#6b7280", ls="--", lw=1)
        axes[1, j].set_xlabel("sau chuẩn hoá + clip", fontsize=8)
    save(fig, out / "scaling_before_after.png")


def projections(X, labels, out: Path, skip_tsne: bool, perplexity: float, cache: dict):
    from sklearn.decomposition import PCA
    full = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#2a78d6")
    ax.axhline(0.95, color="#6b7280", ls=":", lw=1)
    ax.axvline(k95, color="#6b7280", ls=":", lw=1)
    ax.set(xlabel="số thành phần", ylabel="explained variance tích luỹ", ylim=(0, 1.02),
           title="PCA: %d/%d thành phần đạt 95%%" % (k95, X.shape[1]))
    save(fig, out / "pca_explained_variance.png")
    cache["pca_raw"] = full.transform(X)[:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter_by_label(ax, cache["pca_raw"], labels, "PCA 2D (dữ liệu chuẩn hoá)")
    ax.legend(markerscale=3, fontsize=7)
    save(fig, out / "projection_pca.png")
    if not skip_tsne:
        t0 = time.time()
        cache["tsne_raw"] = tsne_2d(X, perplexity)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        scatter_by_label(ax, cache["tsne_raw"], labels, "t-SNE 2D (perplexity %g)" % perplexity)
        ax.legend(markerscale=3, fontsize=7)
        save(fig, out / "projection_tsne.png")
        log("t-SNE %.1fs" % (time.time() - t0))
    try:
        import umap
    except ImportError:
        return log("[skip] UMAP: chưa cài umap-learn")
    P = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter_by_label(ax, P, labels, "UMAP 2D")
    ax.legend(markerscale=3, fontsize=7)
    save(fig, out / "projection_umap.png")


# ---------- Phần 2 ----------
def latent_space(X, labels, models_dir: Path, out: Path, skip_tsne: bool, perplexity: float, cache: dict):
    """Latent AE (encoder) và VAE (μ) trên cùng mẫu với hình chiếu dữ liệu gốc + silhouette."""
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    from train.dl_models import encode_latent, load_checkpoint
    panels = [("raw", "Dữ liệu gốc (chuẩn hoá)", X)]
    for kind, key, title in (("ae", "AE latent", "Latent AE"), ("vae", "VAE latent (mu)", "Latent VAE (μ)")):
        p = models_dir / ("%s.pt" % kind)
        if not p.exists():
            log("[skip] %s: không có %s" % (key, p))
            continue
        _, model = load_checkpoint(p)
        with torch.no_grad():
            Z = encode_latent(model, torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()
        panels.append((key, title, Z))
    if len(panels) == 1:
        return log("[skip] latent: không có ae.pt / vae.pt trong %s" % models_dir)

    def figure(proj_key, project, name, method):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
        for ax, (key, title, M) in zip(axes[0], panels):
            P = cache[proj_key] if key == "raw" and proj_key in cache else project(M)
            scatter_by_label(ax, P, labels, "%s — %s (%d chiều)" % (title, method, M.shape[1]))
        axes[0, -1].legend(markerscale=3, fontsize=7, loc="best")
        save(fig, out / name)

    figure("pca_raw", lambda M: PCA(n_components=2, random_state=SEED).fit_transform(M), "latent_pca.png", "PCA")
    if not skip_tsne:
        figure("tsne_raw", lambda M: tsne_2d(M, perplexity), "latent_tsne.png", "t-SNE")
    y = binarize(labels)
    if len(np.unique(y)) < 2:
        return log("[skip] silhouette: mẫu chỉ có một lớp")
    rows = [{"representation": key, "dim": int(M.shape[1]),
             "silhouette": float(silhouette_score(M, y, sample_size=min(5000, len(M)), random_state=SEED))}
            for key, _, M in panels]
    pd.DataFrame(rows).to_csv(out / "silhouette.csv", index=False)
    log("wrote silhouette.csv")


# ---------- Tổng hợp ----------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(("%.4f" % v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(out: Path, timings: dict):
    lines = ["# EDA — CSE-CIC-IDS2018", "", "## Hình", ""]
    for name, desc in DESCRIPTIONS.items():
        if (out / name).exists():
            lines.append("- `%s` — %s" % (name, desc))
    if (out / "label_counts.csv").exists():
        lc = pd.read_csv(out / "label_counts.csv").set_index("label")
        rows = []
        for split in lc.columns:
            b = int(lc.loc[BENIGN, split]) if BENIGN in lc.index else 0
            a = int(lc[split].sum()) - b
            rows.append({"split": split, "benign": b, "attack": a, "attack_ratio": a / max(a + b, 1)})
        lines += ["", "## Benign / Attack theo tập", "", md_table(pd.DataFrame(rows))]
    if (out / "cleaning_steps.csv").exists():
        lines += ["", "## Hiệu quả làm sạch", "", md_table(pd.read_csv(out / "cleaning_steps.csv"))]
    if (out / "high_corr_pairs.csv").exists():
        n = len(pd.read_csv(out / "high_corr_pairs.csv"))
        lines += ["", "## Tương quan", "",
                  "Số cặp feature có |r| > %.2f trên Benign train: %d (chi tiết: `high_corr_pairs.csv`)." % (CORR_THRESHOLD, n)]
    if (out / "silhouette.csv").exists():
        lines += ["", "## Silhouette theo nhãn nhị phân", "", md_table(pd.read_csv(out / "silhouette.csv"))]
    lines += ["", "## Thời gian chạy lần gần nhất (s)", "", md_table(pd.DataFrame([timings]))]
    (out / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote eda_summary.md")


def fake_stats(data_dir: Path) -> dict:
    """Chỉ dùng cho --smoke: stats giả có cùng cấu trúc với preprocess_stats.json."""
    ld = {s: {k: int(v) for k, v in pd.read_parquet(data_dir / s)[LABEL_COL].value_counts().items()}
          for s in ("train", "val", "test")}
    n = sum(sum(v.values()) for v in ld.values())
    return {"label_distribution": ld, "cleaning": [
        {"step": "read_csv", "rows": n + 13, "rows_removed": 0, "cols": 18},
        {"step": "drop_repeated_headers", "rows": n + 10, "rows_removed": 3, "cols": 18},
        {"step": "drop_nan_inf", "rows": n + 4, "rows_removed": 6, "cols": 18},
        {"step": "drop_constant_cols", "rows": n + 4, "rows_removed": 0, "cols": 17},
        {"step": "drop_duplicates", "rows": n, "rows_removed": 4, "cols": 17}]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None, help="mặc định data.sample_local_dir")
    p.add_argument("--scaler", default=None, help="mặc định models.scaler")
    p.add_argument("--stats", default="results/preprocess_stats.json")
    p.add_argument("--out", default="reports/eda")
    p.add_argument("--per-class", type=int, default=2000, help="số dòng tối đa mỗi nhãn cho phép chiếu 2D")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--max-rows", type=int, default=200000, help="số dòng train tối đa cho KS/tương quan/chuẩn hoá")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--skip-tsne", action="store_true")
    p.add_argument("--latent", action="store_true", help="phần 2: không gian latent AE/VAE + silhouette")
    p.add_argument("--no-part1", action="store_true", help="bỏ mục 1-6 (dùng khi chỉ chạy thêm phần 2)")
    p.add_argument("--models-dir", default="models", help="chứa ae.pt, vae.pt")
    p.add_argument("--smoke", action="store_true", help="dữ liệu giả; với --latent sẽ train AE/VAE 2 epoch")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data_dir = a.data_dir or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    return a


def prepare_smoke(a):
    from common.fake import write_fake_dataset
    tmp = Path(tempfile.mkdtemp(prefix="eda_smoke_"))
    write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
    (tmp / "stats.json").write_text(json.dumps(fake_stats(tmp / "data")), encoding="utf-8")
    a.data_dir, a.scaler, a.stats = str(tmp / "data"), str(tmp / "scaler.json"), str(tmp / "stats.json")
    if a.latent:
        from train import train_dl
        for kind in ("ae", "vae"):
            train_dl.main(["--model", kind, "--data-dir", a.data_dir, "--scaler", a.scaler, "--epochs", "2",
                           "--cpu", "--no-export", "--results-dir", str(tmp / "results"), "--models-dir", str(tmp / "models")])
        a.models_dir = str(tmp / "models")


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.smoke:
        prepare_smoke(a)
    apply_style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timings, cache = {}, {}
    t = time.time()
    scaler = load_scaler(a.scaler)
    stats = json.loads(Path(a.stats).read_text(encoding="utf-8")) if Path(a.stats).exists() else None
    train = read_split_frame(a.data_dir, "train", cap=a.max_rows)
    sample = stratified_sample(train, a.per_class)
    X = frame_to_matrix(sample, scaler)
    labels = sample[LABEL_COL].astype(str).to_numpy()
    timings["load"] = round(time.time() - t, 1)
    log("train %d dòng, mẫu phân tầng %d dòng, %d feature" % (len(train), len(sample), len(scaler.features)))
    if not a.no_part1:
        steps = [
            ("label_distribution", lambda: label_distribution(stats, out)),
            ("cleaning", lambda: cleaning_steps(stats, out)),
            ("ks", lambda: cache.__setitem__("top", ks_analysis(train, scaler.features, a.top_k, sample, out))),
            ("correlation", lambda: correlation(train, scaler.features, out)),
            ("scaling", lambda: scaling(train, scaler, cache["top"], out)),
            ("projections", lambda: projections(X, labels, out, a.skip_tsne, a.perplexity, cache)),
        ]
        for name, fn in steps:
            t = time.time()
            fn()
            timings[name] = round(time.time() - t, 1)
    if a.latent:
        t = time.time()
        latent_space(X, labels, Path(a.models_dir), out, a.skip_tsne, a.perplexity, cache)
        timings["latent"] = round(time.time() - t, 1)
    write_summary(out, timings)
    return timings


if __name__ == "__main__":
    main()
