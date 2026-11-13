"""Gộp metric + vẽ hình cho báo cáo.  python analysis/make_report_assets.py [--results-dir results --reports-dir reports]

Đọc: baselines_metrics.csv, dl_*_metrics.csv, *_per_attack.csv, dl_*_history.csv, scores_*.npz, ablation.csv,
holdout/<attack>/ (holdout.json + metric/per-attack của model supervised train KHÔNG có loại tấn công đó).
Thiếu input nào thì in [skip] và bỏ qua hình đó.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.style import BENIGN_COLOR, CATEGORICAL, DPI, FAMILY_COLORS, apply_style  # noqa: E402

DL_KINDS = {"ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}
DL_NAMES = set(DL_KINDS.values())
AE_FAMILY = ["DeepAutoencoder", "VariationalAE", "DenoisingAE"]
MAIN_MODEL = "DeepAutoencoder"
ATTACK_COLOR = "#eb6834"
TABLE_COLS = ["model", "family", "group", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
              "threshold", "train_time_s", "infer_ms_per_1k", "n_params"]
# Thứ tự cố định theo model (không theo thứ hạng). Mỗi hình gán màu theo thứ tự này cho các model CÓ trong hình
# (tối đa 8 -> không bao giờ trùng màu); nhóm AE luôn đứng đầu nên giữ cùng màu ở mọi hình.
MODEL_ORDER = ["DeepAutoencoder", "VariationalAE", "DenoisingAE", "DNN_Classifier", "LightGBM", "RandomForest",
               "LogisticRegression", "PCA", "IsolationForest", "OneClassSVM"]


def order_key(name: str) -> int:
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)


def model_colors(names) -> dict:
    names = sorted(set(names), key=order_key)
    if len(names) > len(CATEGORICAL):
        raise ValueError("quá %d model trong một hình -> tách hình" % len(CATEGORICAL))
    return {n: CATEGORICAL[i] for i, n in enumerate(names)}


def skip(msg):
    print("[skip] " + msg)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("[ok] " + str(path))


def load_comparison(res: Path) -> pd.DataFrame:
    frames = []
    if (res / "baselines_metrics.csv").exists():
        frames.append(pd.read_csv(res / "baselines_metrics.csv"))
    frames += [pd.read_csv(p) for p in sorted(res.glob("dl_*_metrics.csv"))]
    if not frames:
        raise FileNotFoundError("không có file metric trong %s" % res)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.insert(1, "family", np.where(df["model"].isin(DL_NAMES), "DL", "ML"))
    return df.sort_values(["group", "f1"], ascending=[False, False]).reset_index(drop=True)


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return "%d" % v if float(v).is_integer() and abs(v) >= 1000 else "%.4f" % v
    return str(v)


def to_markdown(df: pd.DataFrame, cols, bold_model=MAIN_MODEL) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [fmt(r[c]) for c in cols]
        if r.get("model") == bold_model:
            cells = ["**%s**" % c if c else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_comparison(df, rep: Path):
    cols = [c for c in TABLE_COLS if c in df.columns]
    df[cols].to_csv(rep / "model_comparison.csv", index=False)
    (rep / "model_comparison.md").write_text(to_markdown(df, cols), encoding="utf-8")


def read_per_attack(res: Path) -> pd.DataFrame:
    files = sorted(res.glob("baselines_per_attack*.csv")) + sorted(res.glob("dl_*_per_attack.csv"))
    frames = [pd.read_csv(f) for f in files if f.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model", "attack", "recall", "n"])


def per_attack_table(res: Path, rep: Path, comp: pd.DataFrame):
    pa = read_per_attack(res)
    if pa.empty:
        return skip("per-attack")
    piv = pa.pivot_table(index="model", columns="attack", values="recall")
    order = [m for m in comp["model"] if m in piv.index] + [m for m in piv.index if m not in set(comp["model"])]
    piv = piv.loc[order]
    n = pa.groupby("attack")["n"].max()
    piv.to_csv(rep / "per_attack_recall.csv")
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(["%s\n(n=%d)" % (c, n.get(c, 0)) for c in piv.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "#111827")
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Recall theo loại tấn công (tập test)")
    save(fig, rep / "per_attack_recall_heatmap.png")


def load_scores(res: Path, name: str):
    p = res / ("scores_%s.npz" % name)
    return np.load(p) if p.exists() else None


def plot_roc_pr(res: Path, models, title, path):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    avail = sorted([m for m in models if load_scores(res, m) is not None], key=order_key)
    if not avail:
        return skip(title)
    colors = model_colors(avail)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in avail:
        d = load_scores(res, name)
        y, s = d["y"], d["score"]
        lw = 2.4 if name == MAIN_MODEL else 1.4
        c = colors[name]
        fpr, tpr, _ = roc_curve(y, s)
        a1.plot(fpr, tpr, lw=lw, color=c, label="%s (AUC %.3f)" % (name, roc_auc_score(y, s)))
        p, r, _ = precision_recall_curve(y, s)
        a2.plot(r, p, lw=lw, color=c, label="%s (AP %.3f)" % (name, average_precision_score(y, s)))
    a1.plot([0, 1], [0, 1], ls=":", color="#6b7280", lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall", ylim=(0, 1.02))
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, path)


def plot_score_hist(res: Path, path):
    names = [n for n in AE_FAMILY if load_scores(res, n) is not None]
    if not names:
        return skip("score histogram")
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = load_scores(res, name)
        y, s = d["y"], np.clip(d["score"].astype(float), 1e-8, None)
        bins = np.logspace(np.log10(s.min()), np.log10(s.max()) + 1e-9, 60)
        for cls, color, lab in ((0, BENIGN_COLOR, "Benign"), (1, ATTACK_COLOR, "Attack")):
            v = s[y == cls]
            if len(v):     # tỷ lệ % trong từng lớp -> hai lớp so sánh được dù số lượng chênh nhau
                ax.hist(v, bins=bins, weights=np.full(len(v), 100.0 / len(v)), histtype="stepfilled",
                        alpha=0.55, color=color, label="%s (n=%d)" % (lab, len(v)))
        ax.axvline(float(d["thr"]), color="#111827", ls="--", lw=1.2, label="ngưỡng")
        ax.set_xscale("log")
        ax.set(title=name, xlabel="reconstruction error (log)", ylabel="% số flow trong lớp")
        ax.legend(fontsize=8)
    save(fig, path)


def plot_loss_curves(res: Path, path, vae_path):
    hist = {k: res / ("dl_%s_history.csv" % k) for k in DL_KINDS}
    if not any(p.exists() for p in hist.values()):
        return skip("loss curves")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (kind, p) in zip(axes.flat, hist.items()):
        if not p.exists():
            ax.axis("off")
            continue
        h = pd.read_csv(p)
        ax.plot(h["epoch"], h["train_loss"], color=CATEGORICAL[0], label="train")
        ax.plot(h["epoch"], h["val_loss"], color=CATEGORICAL[1], label="val")
        ax.set(title=DL_KINDS[kind], xlabel="epoch", ylabel="loss")
        if kind != "mlp":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    save(fig, path)
    if hist["vae"].exists():
        h = pd.read_csv(hist["vae"])
        if {"train_recon", "train_kl"} <= set(h.columns):
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
            for ax, col, title in ((a1, "recon", "VAE reconstruction loss"), (a2, "kl", "VAE KL")):
                ax.plot(h["epoch"], h["train_" + col], color=CATEGORICAL[0], label="train")
                ax.plot(h["epoch"], h["val_" + col], color=CATEGORICAL[1], label="val")
                ax.set(title=title, xlabel="epoch")
                ax.legend(fontsize=8)
            save(fig, vae_path)


def plot_f1_fpr(df: pd.DataFrame, path):
    order = df.sort_values("f1")
    colors = [FAMILY_COLORS[f] for f in order["family"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(order) + 1.8), sharey=True)
    a1.barh(order["model"], order["f1"], color=colors, height=0.7)
    a1.set(xlabel="F1", xlim=(0, 1))
    a2.barh(order["model"], order["fpr"], color=colors, height=0.7)
    a2.set(xlabel="FPR")
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    for y, (f1, fpr) in enumerate(zip(order["f1"], order["fpr"])):
        a1.text(min(f1 + 0.01, 0.9), y, "%.3f" % f1, va="center", fontsize=7, color="#374151")
        a2.text(fpr, y, " %.4f" % fpr, va="center", fontsize=7, color="#374151")
    a1.legend(handles=[Patch(color=c, label=k) for k, c in FAMILY_COLORS.items()], loc="lower right")
    save(fig, path)


def plot_ablation(res: Path, rep: Path):
    p = res / "ablation.csv"
    if not p.exists():
        return skip("ablation")
    ab = pd.read_csv(p)

    def f1_fpr(sub, xlabels, title, xlabel, path):
        """Hai biểu đồ cạnh nhau (F1 | PR-AUC), cùng thang 0–1 — không dùng hai trục y.
        FPR bị ngưỡng p99 ghim ~1% nên chỉ ghi chú, không vẽ (vẽ sẽ phóng đại nhiễu)."""
        x = np.arange(len(sub))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, col, name in zip(axes, ("f1", "pr_auc"), ("F1 (ngưỡng p99)", "PR-AUC (không phụ thuộc ngưỡng)")):
            ax.plot(x, sub[col], marker="o", color=CATEGORICAL[0])
            for xi, v in zip(x, sub[col]):
                ax.annotate("%.3f" % v, (xi, v), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color="#374151")
            ax.set(title=name, ylim=(0, 1.05))
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.set_xlabel(xlabel)
        ep = int(sub["max_epochs"].max()) if "max_epochs" in sub else None
        fig.suptitle(title + (" (tối đa %d epoch)" % ep if ep else "")
                     + "\nFPR %.2f–%.2f%% ở mọi cấu hình (ngưỡng = p99 score val Benign)"
                     % (100 * sub["fpr"].min(), 100 * sub["fpr"].max()), fontsize=11)
        save(fig, path)

    lat = ab[ab["experiment"] == "latent"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(lat):
        f1_fpr(lat, ["%d" % v for v in lat["v"]], "AE: kích thước latent", "latent", rep / "ablation_latent.png")
    dep = ab[ab["experiment"] == "depth"].sort_values("n_params")
    if len(dep):
        f1_fpr(dep, ["%s\n(%s tham số)" % (v, format(int(n), ",")) for v, n in zip(dep["value"], dep["n_params"])],
               "AE: độ sâu", "tầng ẩn encoder", rep / "ablation_depth.png")
    beta = ab[ab["experiment"] == "vae_beta"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(beta):
        f1_fpr(beta, ["%g" % v for v in beta["v"]], "VAE: β", "β", rep / "ablation_vae_beta.png")
    thr = ab[ab["experiment"] == "threshold"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(thr):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(thr))
        for i, col in enumerate(("precision", "recall", "f1", "fpr")):
            ax.plot(x, thr[col], marker="o", color=CATEGORICAL[i], label=col)
        ax.set_xticks(x)
        ax.set_xticklabels(["p%g" % v for v in thr["v"]])
        ax.set(xlabel="percentile score val Benign dùng làm ngưỡng", ylim=(0, 1.02), title="AE: chọn ngưỡng")
        ax.legend(fontsize=8)
        save(fig, rep / "ablation_threshold.png")


def holdout_dirs(res: Path) -> list:
    hdir = res / "holdout"
    if (hdir / "holdout.json").exists():
        return [hdir]
    return sorted(p.parent for p in hdir.glob("*/holdout.json"))


def holdout_table(res: Path, rep: Path):
    dirs = holdout_dirs(res)
    if not dirs:
        return skip("holdout")
    main_pa = read_per_attack(res)
    rows = []
    for d in dirs:
        attack = json.loads((d / "holdout.json").read_text(encoding="utf-8"))["attack"]
        for name in AE_FAMILY:
            r = main_pa[(main_pa["model"] == name) & (main_pa["attack"] == attack)]
            if len(r):
                rows.append({"model": name, "setting": "unsupervised (train chỉ Benign)", "attack": attack,
                             "recall": float(r["recall"].iloc[0]), "n": int(r["n"].iloc[0])})
        sup = {"DNN_Classifier"}
        for f in d.glob("baselines_metrics*.csv"):
            bm = pd.read_csv(f)
            sup |= set(bm.loc[bm["group"] == "supervised", "model"])
        hpa = read_per_attack(d)
        for _, r in hpa[(hpa["attack"] == attack) & hpa["model"].isin(sup)].iterrows():
            rows.append({"model": r["model"], "setting": "supervised, holdout khỏi train", "attack": attack,
                         "recall": float(r["recall"]), "n": int(r["n"])})
    df = pd.DataFrame(rows, columns=["model", "setting", "attack", "recall", "n"])
    df.to_csv(rep / "holdout_recall.csv", index=False)
    piv = df.pivot_table(index=["model", "setting"], columns="attack", values="recall").reset_index()
    piv["_o"] = piv["model"].map(order_key)
    piv = piv.sort_values("_o").drop(columns="_o")
    (rep / "holdout_recall.md").write_text(to_markdown(piv, list(piv.columns)), encoding="utf-8")

    attacks = sorted(df["attack"].unique())
    models = list(piv["model"])
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(2.6 * len(attacks) + 3, 4.2))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [df[(df["model"] == m) & (df["attack"] == a)]["recall"].mean() for a in attacks]
        sup = m not in AE_FAMILY
        ax.bar(np.arange(len(attacks)) + (i - (len(models) - 1) / 2) * w, np.nan_to_num(vals), width=w * 0.92,
               color=colors[m], hatch="//" if sup else None, edgecolor="white", linewidth=0.5,
               label=m + (" (supervised, holdout)" if sup else " (unsupervised)"))
    ax.set_xticks(np.arange(len(attacks)))
    ax.set_xticklabels(attacks)
    ax.set(ylabel="Recall trên loại tấn công bị giữ lại", ylim=(0, 1.05),
           title="Tấn công chưa thấy khi train: unsupervised vs supervised")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, rep / "holdout_recall.png")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reports-dir", default="reports")
    a = p.parse_args(argv)
    apply_style()
    res, rep = Path(a.results_dir), Path(a.reports_dir)
    rep.mkdir(parents=True, exist_ok=True)
    comp = load_comparison(res)
    write_comparison(comp, rep)
    per_attack_table(res, rep, comp)
    groups = comp.groupby("group")["model"].apply(list).to_dict()
    plot_roc_pr(res, groups.get("unsupervised", []), "Nhóm unsupervised", rep / "roc_pr_unsupervised.png")
    plot_roc_pr(res, groups.get("supervised", []), "Nhóm supervised", rep / "roc_pr_supervised.png")
    plot_score_hist(res, rep / "score_hist_ae_family.png")
    plot_loss_curves(res, rep / "loss_curves.png", rep / "vae_recon_kl.png")
    plot_f1_fpr(comp, rep / "f1_fpr_bar.png")
    plot_ablation(res, rep)
    holdout_table(res, rep)
    print(to_markdown(comp, [c for c in TABLE_COLS if c in comp.columns]))


if __name__ == "__main__":
    main()
