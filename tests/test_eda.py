import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
from analysis import eda  # noqa: E402
from common.data import LABEL_COL  # noqa: E402
from common.fake import FAKE_ATTACKS, make_fake_flows  # noqa: E402

PART1 = ["label_distribution.png", "label_counts.csv", "cleaning_steps.csv", "ks_table.csv", "top_features_box.png",
         "correlation_heatmap.png", "high_corr_pairs.csv", "scaling_before_after.png", "clip_rate.csv", "clip_rate.png",
         "pca_explained_variance.png", "projection_pca.png", "projection_tsne.png", "eda_summary.md"]


def test_stratified_sample():
    df = make_fake_flows(2000)
    s = eda.stratified_sample(df, per_class=100)
    counts = s[LABEL_COL].value_counts()
    assert set(counts.index) == {"Benign", *FAKE_ATTACKS} and (counts == 100).all()
    pd.testing.assert_frame_equal(s, eda.stratified_sample(df, per_class=100))   # seed cố định


def test_part1_smoke(tmp_path):
    out = tmp_path / "eda"
    eda.main(["--smoke", "--out", str(out), "--per-class", "150"])
    missing = [f for f in PART1 if not (out / f).exists()]
    assert missing == []
    ks = pd.read_csv(out / "ks_table.csv")
    assert list(ks.columns) == ["feature", "ks", "pvalue"] and ks["ks"].is_monotonic_decreasing
    summary = (out / "eda_summary.md").read_text(encoding="utf-8")
    for f in PART1:
        if f.endswith(".png"):
            assert "`%s`" % f in summary
    assert not (out / "latent_pca.png").exists()


def test_missing_stats_is_skipped(tmp_path):
    from common.fake import write_fake_dataset
    write_fake_dataset(tmp_path / "data", n=3000, scaler_path=tmp_path / "scaler.json")
    out = tmp_path / "eda"
    eda.main(["--data-dir", str(tmp_path / "data"), "--scaler", str(tmp_path / "scaler.json"),
              "--stats", str(tmp_path / "none.json"), "--out", str(out), "--per-class", "100", "--skip-tsne"])
    assert not (out / "label_distribution.png").exists() and (out / "projection_pca.png").exists()
    assert not (out / "projection_tsne.png").exists()


def test_latent_smoke(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "eda"
    eda.main(["--smoke", "--latent", "--out", str(out), "--per-class", "150"])
    for f in ("latent_pca.png", "latent_tsne.png", "silhouette.csv"):
        assert (out / f).exists(), f
    sil = pd.read_csv(out / "silhouette.csv")
    assert list(sil.columns) == ["representation", "dim", "silhouette"]
    assert sil["representation"].tolist() == ["raw", "AE latent", "VAE latent (mu)"]
    assert sil["silhouette"].between(-1, 1).all()
    assert "Silhouette" in (out / "eda_summary.md").read_text(encoding="utf-8")


def test_latent_only_keeps_part1_descriptions(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "eda"
    eda.main(["--smoke", "--out", str(out), "--per-class", "100", "--skip-tsne"])
    eda.main(["--smoke", "--latent", "--no-part1", "--out", str(out), "--per-class", "100", "--skip-tsne"])
    summary = (out / "eda_summary.md").read_text(encoding="utf-8")
    assert "`projection_pca.png`" in summary and "`latent_pca.png`" in summary
