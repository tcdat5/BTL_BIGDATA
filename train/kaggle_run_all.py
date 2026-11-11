"""Chạy toàn bộ pipeline Deep Learning trên Kaggle (hoặc cục bộ): preprocess -> EDA -> train -> đánh giá -> đóng gói.

Notebook (train/kaggle_notebook.ipynb) gọi từng stage một ô:
    from train.kaggle_run_all import Runner
    R = Runner(raw_dir=RAW_DIR, work_dir=WORK_DIR, quick=QUICK, file_map=FILE_MAP)
    R.find_data(); R.stage("preprocess"); R.stage("ae"); ...; R.package()
CLI (chạy hết một lượt):
    python train/kaggle_run_all.py --raw-dir /kaggle/input/<dataset> --work-dir /kaggle/working [--quick]

Stage lỗi: in traceback rồi chạy tiếp; riêng preprocess và ae lỗi thì dừng (raise).
Stage nào chưa có script (vd eda/ablation chưa viết) thì bỏ qua và ghi "skipped".
QUICK: 1 file CSV, train 50k / val 20k / test 20k, 3 epoch — để chạy thử < 15 phút.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402

CRITICAL = ("preprocess", "ae")
STAGES = ["preprocess", "eda", "ae", "mlp", "vae", "dae", "baselines", "ablation", "holdout",
          "eda_latent", "report"]
HOLDOUT_ATTACKS = ["Bot", "DDOS attack-HOIC", "SSH-Bruteforce"]
QUICK_SIZES = {"train": 50000, "val": 20000, "test": 20000}
QUICK_EPOCHS = 3


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (AttributeError, ValueError, OSError):
        return 8.0


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def date_tokens(name: str) -> set:
    """'Wednesday-14-02-2018_...' và '02-14-2018.csv' cùng cho {('14','02','2018')}."""
    out = set()
    for a, b, y in re.findall(r"(\d{2})-(\d{2})-(\d{4})", name):
        out.add(tuple(sorted((a, b))) + (y,))
    return out


class Runner:
    def __init__(self, raw_dir, work_dir, quick=False, file_map=None, config=None, python=sys.executable,
                 holdout_attacks=None):
        # tuyệt đối: script con chạy với cwd=ROOT, đường dẫn tương đối sẽ sai
        self.raw_dir = Path(raw_dir).resolve()
        self.work = Path(work_dir).resolve()
        self.quick = quick
        self.file_map = dict(file_map or {})
        self.cfg = C.load_config(config)
        self.py = python
        self.holdout_attacks = list(holdout_attacks or HOLDOUT_ATTACKS)
        self.timings = []
        self.paths = {}
        for k, sub in (("processed", "data/processed"), ("sample", "data/kaggle_sample"),
                       ("demo", "data/demo_test.parquet"), ("models", "models"), ("results", "results"),
                       ("reports", "reports")):
            self.paths[k] = self.work / sub
        self.paths["scaler"] = self.paths["models"] / "scaler.json"
        self.paths["stats"] = self.paths["results"] / "preprocess_stats.json"
        self.files = None

    # ------------------------------------------------------------ helpers
    def run_cmd(self, args) -> None:
        """Chạy script con, in output từng dòng (để hiện trong ô notebook)."""
        cmd = [self.py] + [str(a) for a in args]
        print("$ " + " ".join(cmd), flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            print(line, end="", flush=True)
        if p.wait() != 0:
            raise RuntimeError("lệnh lỗi (exit %d): %s" % (p.returncode, " ".join(cmd)))

    def script(self, rel: str) -> Path:
        return ROOT / rel

    def common_train_args(self):
        args = ["--data-dir", self.paths["sample"], "--scaler", self.paths["scaler"],
                "--models-dir", self.paths["models"], "--results-dir", self.paths["results"]]
        if self.quick:
            args += ["--epochs", QUICK_EPOCHS, "--train-cap", QUICK_SIZES["train"], "--test-cap", QUICK_SIZES["test"]]
        return args

    def stage(self, name: str):
        """Chạy một stage, ghi thời gian; lỗi -> in traceback và chạy tiếp (trừ CRITICAL)."""
        fn = getattr(self, "st_" + name)
        print("\n" + "=" * 70 + "\n[stage] %s  (%s)\n" % (name, time.strftime("%H:%M:%S")) + "=" * 70, flush=True)
        t0 = time.time()
        status = "ok"
        try:
            if fn() == "skipped":
                status = "skipped"
        except Exception:
            status = "FAILED"
            traceback.print_exc()
        sec = round(time.time() - t0, 1)
        self.timings.append({"stage": name, "status": status, "seconds": sec})
        print("[stage] %s -> %s (%.1fs)" % (name, status, sec), flush=True)
        if status == "FAILED" and name in CRITICAL:
            self.write_timing()
            raise RuntimeError("stage '%s' lỗi -> dừng pipeline (xem traceback ở trên)" % name)
        return status

    def run_all(self, stages=None):
        self.find_data()
        for s in stages or STAGES:
            self.stage(s)
        return self.package()

    # ------------------------------------------------------------ data
    def find_data(self) -> list:
        """Tìm các CSV trong data.raw_files dưới raw_dir (đệ quy). Báo lỗi rõ nếu thiếu."""
        if not self.raw_dir.exists():
            # Kaggle đổi chỗ mount dataset theo thời gian (vd /kaggle/input/datasets/<owner>/<slug>) -> tìm khắp /kaggle/input
            fallback = Path("/kaggle/input")
            if not fallback.exists():
                raise FileNotFoundError("RAW_DIR không tồn tại: %s" % self.raw_dir)
            print("[warn] RAW_DIR %s không tồn tại -> tìm trong %s" % (self.raw_dir, fallback))
            self.raw_dir = fallback
        found = sorted(p for p in self.raw_dir.rglob("*") if p.suffix.lower() == ".csv")
        print("CSV tìm thấy dưới %s (%d file):" % (self.raw_dir, len(found)))
        for p in found:
            print("  %8.1f MB  %s" % (p.stat().st_size / 1e6, p.relative_to(self.raw_dir)))
        wanted = list(C.get(self.cfg, "data.raw_files"))
        if self.quick:
            wanted = wanted[:1]
        by_name = {p.name: p for p in found}
        resolved, missing = [], []
        for w in wanted:
            target = self.file_map.get(w, w)
            if target in by_name:
                resolved.append(by_name[target])
                continue
            cands = [p for p in found if date_tokens(p.name) & date_tokens(w)]
            if len(cands) == 1:
                print("[map] %s -> %s (khớp theo ngày; đặt FILE_MAP nếu sai)" % (w, cands[0].name))
                resolved.append(cands[0])
            else:
                missing.append((w, [c.name for c in cands]))
        if missing:
            msg = "\n".join("  - %s (ứng viên: %s)" % (w, c or "không có") for w, c in missing)
            raise FileNotFoundError("Thiếu file CSV trong data.raw_files:\n%s\n"
                                    "-> Đặt FILE_MAP = {'<tên trong config>': '<tên file trên Kaggle>'}" % msg)
        self.files = resolved
        print("Dùng %d file: %s" % (len(resolved), [p.name for p in resolved]))
        return resolved

    def attacks_in_train(self) -> set:
        if not self.paths["stats"].exists():
            return set()
        st = json.loads(self.paths["stats"].read_text(encoding="utf-8"))
        return set(st.get("label_distribution", {}).get("train", {})) - {"Benign"}

    # ------------------------------------------------------------ stages
    def st_preprocess(self):
        if self.files is None:
            self.find_data()
        sizes = QUICK_SIZES if self.quick else C.get(self.cfg, "data.sample_sizes")
        mem = "%dg" % max(1, int(ram_gb() * 0.7))
        self.run_cmd([self.script("preprocess/spark_preprocess.py"), "--master", "local[*]",
                      "--driver-memory", mem, "--files", *[str(p) for p in self.files],
                      "--input-dir", self.raw_dir, "--out", self.paths["processed"],
                      "--export-sample", self.paths["sample"],
                      "--sample-train", sizes["train"], "--sample-val", sizes["val"], "--sample-test", sizes["test"],
                      "--demo-out", self.paths["demo"], "--scaler-out", self.paths["scaler"],
                      "--stats-out", self.paths["stats"]])

    def _dl(self, kind):
        self.run_cmd([self.script("train/train_dl.py"), "--model", kind] + self.common_train_args())

    def st_ae(self):
        self._dl("ae")

    def st_mlp(self):
        self._dl("mlp")

    def st_vae(self):
        self._dl("vae")

    def st_dae(self):
        self._dl("dae")

    def _baseline_args(self):
        args = ["--data", self.paths["sample"], "--scaler", self.paths["scaler"]]
        if self.quick:
            args += ["--train-cap", QUICK_SIZES["train"], "--test-cap", QUICK_SIZES["test"]]
        return args

    def st_baselines(self):
        self.run_cmd([self.script("train/baselines.py")] + self._baseline_args() + ["--out", self.paths["results"]])

    def st_eda(self):
        if not self.script("analysis/eda.py").exists():
            print("[skip] analysis/eda.py chưa có")
            return "skipped"
        args = ["--data-dir", self.paths["sample"], "--scaler", self.paths["scaler"], "--stats", self.paths["stats"],
                "--out", self.paths["reports"] / "eda"]
        if self.quick:
            args += ["--per-class", 300]
        self.run_cmd([self.script("analysis/eda.py")] + args)

    def st_ablation(self):
        if not self.script("train/run_ablation.py").exists():
            print("[skip] train/run_ablation.py chưa có")
            return "skipped"
        args = ["--data-dir", self.paths["sample"], "--scaler", self.paths["scaler"],
                "--base-results-dir", self.paths["results"], "--out-root", self.paths["results"] / "ablation",
                "--out-csv", self.paths["results"] / "ablation.csv",
                "--only", "latent,threshold,depth" if self.quick else "latent,threshold,depth,vae_beta"]
        if self.quick:
            args += ["--epochs", QUICK_EPOCHS, "--test-cap", QUICK_SIZES["test"]]
        self.run_cmd([self.script("train/run_ablation.py")] + args)

    def st_holdout(self):
        present = self.attacks_in_train()
        done = 0
        for att in self.holdout_attacks:
            if att not in present:
                print("[warn] holdout '%s': không có trong train của dữ liệu đã chọn -> bỏ qua" % att)
                continue
            out = self.paths["results"] / "holdout" / slug(att)
            self.run_cmd([self.script("train/train_dl.py"), "--model", "mlp", "--holdout-attack", att, "--no-export"]
                         + self.common_train_args()[:4] + ["--models-dir", out / "models", "--results-dir", out]
                         + (["--epochs", QUICK_EPOCHS, "--train-cap", QUICK_SIZES["train"],
                             "--test-cap", QUICK_SIZES["test"]] if self.quick else []))
            self.run_cmd([self.script("train/baselines.py")] + self._baseline_args()
                         + ["--holdout-attack", att, "--out", out])
            done += 1
        return "ok" if done else "skipped"

    def st_eda_latent(self):
        if not self.script("analysis/eda.py").exists():
            print("[skip] analysis/eda.py chưa có")
            return "skipped"
        args = ["--data-dir", self.paths["sample"], "--scaler", self.paths["scaler"], "--latent", "--no-part1",
                "--models-dir", self.paths["models"], "--out", self.paths["reports"] / "eda"]
        if self.quick:
            args += ["--per-class", 300]
        self.run_cmd([self.script("analysis/eda.py")] + args)

    def st_report(self):
        if not self.script("analysis/make_report_assets.py").exists():
            print("[skip] analysis/make_report_assets.py chưa có")
            return "skipped"
        self.run_cmd([self.script("analysis/make_report_assets.py"), "--results-dir", self.paths["results"],
                      "--reports-dir", self.paths["reports"]])

    # ------------------------------------------------------------ output
    def write_timing(self) -> pd.DataFrame:
        df = pd.DataFrame(self.timings, columns=["stage", "status", "seconds"])
        self.paths["results"].mkdir(parents=True, exist_ok=True)
        df.to_csv(self.paths["results"] / "kaggle_timing.csv", index=False)
        return df

    def package(self) -> Path:
        """Gói models/ results/ reports/ data/demo_test.parquet vào kaggle_outputs.zip, in bảng thời gian."""
        df = self.write_timing()
        zpath = self.work / "kaggle_outputs.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for key in ("models", "results", "reports"):
                base = self.paths[key]
                if base.exists():
                    for f in sorted(base.rglob("*")):
                        if f.is_file():
                            z.write(f, f.relative_to(self.work).as_posix())
            if self.paths["demo"].exists():
                z.write(self.paths["demo"], self.paths["demo"].relative_to(self.work).as_posix())
        print("\nThời gian từng bước:")
        print(df.to_string(index=False))
        print("TỔNG: %.1f phút" % (df["seconds"].sum() / 60))
        print("Đã đóng gói: %s (%.1f MB)" % (zpath, zpath.stat().st_size / 1e6))
        return zpath


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="thư mục chứa CSV CSE-CIC-IDS2018 (tìm đệ quy)")
    p.add_argument("--work-dir", default=".", help="nơi ghi data/ models/ results/ reports/ (Kaggle: /kaggle/working)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--stages", default="all", help="danh sách stage, phân tách dấu phẩy: " + ",".join(STAGES))
    p.add_argument("--file-map", default="{}", help='JSON: {"<tên trong config>": "<tên file thật>"}')
    p.add_argument("--config", default=None)
    a = p.parse_args(argv)
    r = Runner(a.raw_dir, a.work_dir, quick=a.quick, file_map=json.loads(a.file_map), config=a.config)
    stages = STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    return r.run_all(stages)


if __name__ == "__main__":
    main()
