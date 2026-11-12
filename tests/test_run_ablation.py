import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
from train import run_ablation  # noqa: E402


def test_ablation_smoke(tmp_path):
    csv = tmp_path / "ablation.csv"
    run_ablation.main(["--smoke", "--only", "latent,depth,threshold,vae_beta",
                       "--out-root", str(tmp_path / "runs"), "--out-csv", str(csv),
                       "--base-results-dir", str(tmp_path / "no_base")])
    df = pd.read_csv(csv)
    assert list(df.columns) == run_ablation.ABLATION_COLUMNS
    assert df.groupby("experiment").size().to_dict() == {"latent": 4, "depth": 3, "threshold": 5, "vae_beta": 3}
    thr = df[df.experiment == "threshold"].sort_values("value")
    assert thr["fpr"].is_monotonic_decreasing          # ngưỡng cao hơn -> FPR không tăng
    d = df[df.experiment == "depth"].set_index("value")["n_params"]
    assert d["32"] < d["64,32"] < d["128,64,32"]
    # cấu hình mặc định (64,32 / latent 16) chỉ train một lần
    lat16 = df[(df.experiment == "latent") & (df.value.astype(str) == "16")].iloc[0]
    assert lat16["n_params"] == d["64,32"]
    assert len(list((tmp_path / "runs").iterdir())) == 4 + 2 + 3


def test_only_replaces_selected_experiments(tmp_path):
    csv = tmp_path / "ablation.csv"
    pd.DataFrame([{"experiment": "depth", "value": "x", **{c: 0 for c in run_ablation.ABLATION_COLUMNS[2:]}}]) \
        .to_csv(csv, index=False)
    run_ablation.main(["--smoke", "--only", "threshold", "--out-root", str(tmp_path / "runs"),
                       "--out-csv", str(csv), "--base-results-dir", str(tmp_path / "no_base")])
    df = pd.read_csv(csv)
    assert (df.experiment == "depth").sum() == 1 and (df.experiment == "threshold").sum() == 5
