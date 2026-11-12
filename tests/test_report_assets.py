import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
from analysis import make_report_assets as R  # noqa: E402
from common.metrics import METRIC_COLUMNS  # noqa: E402

UNSUP = ["DeepAutoencoder", "VariationalAE", "DenoisingAE", "PCA"]
SUP = ["DNN_Classifier", "RandomForest"]
KIND = {"DeepAutoencoder": "ae", "VariationalAE": "vae", "DenoisingAE": "dae", "DNN_Classifier": "mlp"}


def _metric_row(name, group, f1):
    return {"model": name, "group": group, "precision": f1, "recall": f1, "f1": f1, "fpr": 0.01,
            "roc_auc": 0.9, "pr_auc": 0.8, "threshold": 0.5, "train_time_s": 1.0, "infer_ms_per_1k": 0.1}


def _per_attack(name):
    return pd.DataFrame({"model": name, "attack": ["Bot", "DDOS attack-HOIC"], "recall": [0.9, 0.4], "n": [10, 20]})


def _holdout(h, attack):
    h.mkdir(parents=True)
    (h / "holdout.json").write_text(json.dumps({"attack": attack}))
    _per_attack("DNN_Classifier").to_csv(h / "dl_mlp_per_attack.csv", index=False)
    tag = attack.replace(" ", "_")
    pd.concat([_per_attack("PCA"), _per_attack("RandomForest")]).to_csv(
        h / ("baselines_per_attack_holdout-%s.csv" % tag), index=False)
    pd.DataFrame([_metric_row("PCA", "unsupervised", 0.6), _metric_row("RandomForest", "supervised", 0.9)],
                 columns=METRIC_COLUMNS).to_csv(h / ("baselines_metrics_holdout-%s.csv" % tag), index=False)


def _fake_results(res):
    rng = np.random.default_rng(0)
    res.mkdir(parents=True)
    pd.DataFrame([_metric_row("PCA", "unsupervised", 0.6), _metric_row("RandomForest", "supervised", 0.95)],
                 columns=METRIC_COLUMNS).to_csv(res / "baselines_metrics.csv", index=False)
    pd.concat([_per_attack("PCA"), _per_attack("RandomForest")]).to_csv(res / "baselines_per_attack.csv", index=False)
    for i, (name, kind) in enumerate(KIND.items()):
        grp = "supervised" if kind == "mlp" else "unsupervised"
        row = dict(_metric_row(name, grp, 0.7 + i * 0.05), n_params=1000 + i)
        pd.DataFrame([row]).to_csv(res / ("dl_%s_metrics.csv" % kind), index=False)
        _per_attack(name).to_csv(res / ("dl_%s_per_attack.csv" % kind), index=False)
        h = pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [1.0, 0.5, 0.4], "val_loss": [1.1, 0.6, 0.5]})
        if kind == "vae":
            h = h.assign(train_recon=[0.9, 0.4, 0.3], train_kl=[0.1, 0.1, 0.1], val_recon=[1.0, 0.5, 0.4],
                         val_kl=[0.1, 0.1, 0.1])
        h.to_csv(res / ("dl_%s_history.csv" % kind), index=False)
    for name in UNSUP + SUP:
        y = rng.integers(0, 2, 500)
        np.savez(res / ("scores_%s.npz" % name), y=y, score=rng.random(500) + y, thr=np.array(1.0))
    rows = []
    for exp, vals in (("latent", [4, 8, 16, 32]), ("depth", ["32", "64,32", "128,64,32"]),
                      ("threshold", [95, 97, 99, 99.5, 99.9]), ("vae_beta", [0.5, 1.0, 2.0])):
        for j, v in enumerate(vals):
            rows.append({"experiment": exp, "value": v, "precision": 0.8, "recall": 0.7, "f1": 0.75,
                         "fpr": 0.05 - j * 0.01, "roc_auc": 0.9, "pr_auc": 0.8, "n_params": 1000 * (j + 1),
                         "train_time_s": 1.0, "max_epochs": 20})
    pd.DataFrame(rows).to_csv(res / "ablation.csv", index=False)
    _holdout(res / "holdout" / "Bot", "Bot")
    _holdout(res / "holdout" / "DDOS_attack_HOIC", "DDOS attack-HOIC")


EXPECTED = ["model_comparison.csv", "model_comparison.md", "per_attack_recall.csv", "per_attack_recall_heatmap.png",
            "roc_pr_unsupervised.png", "roc_pr_supervised.png", "score_hist_ae_family.png", "loss_curves.png",
            "vae_recon_kl.png", "f1_fpr_bar.png", "ablation_latent.png", "ablation_depth.png",
            "ablation_threshold.png", "ablation_vae_beta.png", "holdout_recall.csv", "holdout_recall.md",
            "holdout_recall.png"]


def test_all_assets(tmp_path):
    res, rep = tmp_path / "results", tmp_path / "reports"
    _fake_results(res)
    R.main(["--results-dir", str(res), "--reports-dir", str(rep)])
    missing = [f for f in EXPECTED if not (rep / f).exists()]
    assert missing == []
    comp = pd.read_csv(rep / "model_comparison.csv")
    assert len(comp) == 6 and set(comp["family"]) == {"DL", "ML"}
    assert comp["group"].tolist()[:4] == ["unsupervised"] * 4
    md = (rep / "model_comparison.md").read_text(encoding="utf-8")
    assert "| **DeepAutoencoder** |" in md and "0.0100" in md
    ho = pd.read_csv(rep / "holdout_recall.csv")
    assert set(ho["model"]) == {"DeepAutoencoder", "VariationalAE", "DenoisingAE", "DNN_Classifier", "RandomForest"}
    assert set(ho["attack"]) == {"Bot", "DDOS attack-HOIC"}
    assert len(ho) == 2 * 5


def test_missing_optional_inputs_are_skipped(tmp_path):
    res, rep = tmp_path / "results", tmp_path / "reports"
    _fake_results(res)
    (res / "ablation.csv").unlink()
    R.main(["--results-dir", str(res), "--reports-dir", str(rep)])
    assert (rep / "model_comparison.md").exists() and not (rep / "ablation_latent.png").exists()
