"""Sinh train/kaggle_notebook.ipynb (giữ notebook là output của code để dễ review/diff).

python train/make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kaggle_notebook.ipynb"

MD_INTRO = """# BTL_BigDeep — Deep Learning pipeline trên Kaggle

Chạy một lượt bằng **Save Version → Save & Run All**. Trước khi chạy:
1. **Add Data**: dataset CSE-CIC-IDS2018 (CSV) — mặc định `RAW_DIR = "/kaggle/input"` tự tìm CSV theo ngày.
2. **Accelerator**: GPU (T4 x2 hoặc P100). **Internet**: On (cần cho git clone / pip / apt).
3. Chạy thử với `QUICK = True` (< 15 phút), xong mới đặt `QUICK = False`.

Kết quả: `/kaggle/working/kaggle_outputs.zip` (models/, results/, reports/, data/demo_test.parquet)."""

CELL_PARAMS = '''# ===== Tham số =====
QUICK = False                      # True: 1 file CSV, train 50k / test 20k, 3 epoch
RAW_DIR = "/kaggle/input"        # thư mục chứa CSV CSE-CIC-IDS2018 (tìm đệ quy, khớp theo ngày)
WORK_DIR = "/kaggle/working"
REPO_URL = "https://github.com/tcdat5/BTL_BIGDEEP.git"   # repo public: git clone
CODE_DATASET = ""                  # repo private: đường dẫn Kaggle Dataset chứa mã nguồn, vd "/kaggle/input/btl-bigdeep-code"
CODE_DIR = "/kaggle/working/BTL_BIGDEEP"
FILE_MAP = {}                      # nếu tên file khác config: {"Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": "02-14-2018.csv"}
# None = chạy tất cả. Chỉ chạy một phần, vd ablation + EDA (khoảng 25 phút):
# STAGES = ["preprocess", "eda", "ae", "vae", "ablation", "eda_latent"]
# Các stage: preprocess, eda, ae, mlp, vae, dae, baselines, ablation, holdout, eda_latent, report
STAGES = None'''

CELL_SETUP = '''# ===== 1. Thiết lập: phần cứng, mã nguồn, package =====
import os, sys, shutil, subprocess, importlib.util
print("CPU:", os.cpu_count(), "| RAM: %.1f GB" % (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3))
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout or "Không có GPU")
if CODE_DATASET:
    shutil.copytree(CODE_DATASET, CODE_DIR, dirs_exist_ok=True)
elif not os.path.exists(CODE_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE_DIR], check=True)
else:
    subprocess.run(["git", "-C", CODE_DIR, "pull", "--ff-only"], check=False)
os.chdir(CODE_DIR); sys.path.insert(0, CODE_DIR)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
need = {"torch": "torch", "onnx": "onnx", "onnxruntime": "onnxruntime", "sklearn": "scikit-learn",
        "lightgbm": "lightgbm", "pyarrow": "pyarrow", "matplotlib": "matplotlib", "yaml": "pyyaml",
        "scipy": "scipy", "pyspark": "pyspark==3.5.1"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)
print("pip install:", missing or "không thiếu gì")'''

CELL_JAVA = '''# ===== 2. Java cho PySpark (Spark 3.5: Java 17; Spark 4: Java 17/21) =====
import glob
def java_ok():
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True)
        print((r.stderr or r.stdout).splitlines()[0]); return r.returncode == 0
    except FileNotFoundError:
        return False
if not java_ok():
    subprocess.run("apt-get update -qq && apt-get install -y -qq openjdk-17-jre-headless", shell=True, check=True)
    java_ok()
if not os.environ.get("JAVA_HOME") and shutil.which("java"):
    os.environ["JAVA_HOME"] = os.path.dirname(os.path.dirname(os.path.realpath(shutil.which("java"))))
print("JAVA_HOME =", os.environ.get("JAVA_HOME"))
import pyspark; print("pyspark", pyspark.__version__, "| python", sys.version.split()[0])'''

CELL_FIND = '''# ===== 3. Tìm dữ liệu =====
from train.kaggle_run_all import Runner
R = Runner(raw_dir=RAW_DIR, work_dir=WORK_DIR, quick=QUICK, file_map=FILE_MAP, only=STAGES)
R.find_data()'''

STAGE_CELLS = [
    ("4. Preprocess (Spark local[*]) — lỗi thì dừng", ["preprocess"]),
    ("5. EDA phần 1 (CPU)", ["eda"]),
    ("6a. Deep Autoencoder (GPU) — lỗi thì dừng", ["ae"]),
    ("6b. DNN classifier", ["mlp"]),
    ("6c. Variational AE", ["vae"]),
    ("6d. Denoising AE", ["dae"]),
    ("7. 6 baseline ML", ["baselines"]),
    ("8. Ablation + holdout (Bot, DDOS attack-HOIC, SSH-Bruteforce)", ["ablation", "holdout"]),
    ("9. EDA latent + bảng/hình báo cáo", ["eda_latent", "report"]),
]

CELL_PACKAGE = '''# ===== 10. Đóng gói output =====
R.package()'''


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build() -> dict:
    cells = [md(MD_INTRO), code(CELL_PARAMS), code(CELL_SETUP), code(CELL_JAVA), code(CELL_FIND)]
    for title, stages in STAGE_CELLS:
        cells.append(code("# ===== %s =====\n" % title + "\n".join('R.stage("%s")' % s for s in stages)))
    cells.append(code(CELL_PACKAGE))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "cells": cells}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
