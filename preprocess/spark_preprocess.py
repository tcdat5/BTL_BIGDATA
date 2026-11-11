"""CSE-CIC-IDS2018 CSV -> Parquet train/val/test + scaler.json (+ bản mẫu Kaggle + demo_test.parquet).

Chính sách giá trị lỗi: dòng có null/NaN/±Infinity ở BẤT KỲ cột feature nào bị bỏ.
Chia tập TẤT ĐỊNH (giống hệt nhau trên Kaggle local[*] và trên cụm, bất kể số partition):
  flow_id = xxhash64(<feature theo thứ tự cố định>, Label)   (duy nhất sau khi gộp dòng trùng)
  bucket  = pmod(flow_id, 100): < 70 train, 70-79 val, >= 80 test
Dòng trùng (cùng feature + Label) gộp làm một, giữ orig_ts nhỏ nhất.

Cụm:    bash scripts/run_preprocess.sh
Kaggle: python preprocess/spark_preprocess.py --master "local[*]" --input-dir /kaggle/input/... \
          --out /kaggle/working/data/processed --export-sample /kaggle/working/data/kaggle_sample \
          --demo-out /kaggle/working/data/demo_test.parquet
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import BENIGN, ID_COL, LABEL_COL, SEED, TS_COL, Scaler  # noqa: E402

ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port"]
RAW_TS = "Timestamp"
SPLITS = (("train", 0, 70), ("val", 70, 80), ("test", 80, 100))
DEMO_MIN_PER_ATTACK = 500


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--master", default=None, help="spark://host:7077 hoặc local[*]; bỏ trống khi spark-submit --master")
    p.add_argument("--input-dir", default=None, help="thư mục CSV, local hoặc hdfs:// (mặc định hdfs.raw_dir)")
    p.add_argument("--files", nargs="+", default=None, help="tên file CSV (mặc định data.raw_files)")
    p.add_argument("--out", default=None, help="thư mục Parquet ra (mặc định hdfs.processed_dir)")
    p.add_argument("--export-sample", default=None, help="thư mục bản mẫu cho Kaggle (cùng FS với --out); bỏ trống = không xuất")
    p.add_argument("--sample-train", type=int, default=None)
    p.add_argument("--sample-val", type=int, default=None)
    p.add_argument("--sample-test", type=int, default=None)
    p.add_argument("--demo-out", default=None, help="file Parquet cục bộ (trên driver) cho dashboard replay; bỏ trống = không xuất")
    p.add_argument("--demo-rows", type=int, default=50000)
    p.add_argument("--scaler-out", default=None, help="mặc định models.scaler")
    p.add_argument("--stats-out", default="results/preprocess_stats.json")
    p.add_argument("--executor-memory", default=None)
    p.add_argument("--driver-memory", default=None)
    p.add_argument("--shuffle-partitions", type=int, default=None)
    p.add_argument("--persist-level", default=None, choices=["MEMORY_AND_DISK", "DISK_ONLY"],
                   help="mặc định spark.persist_level hoặc MEMORY_AND_DISK; slave ít RAM dùng DISK_ONLY")
    p.add_argument("--debug-repartition", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.input_dir = a.input_dir or C.get(cfg, "hdfs.raw_dir")
    a.files = a.files or C.get(cfg, "data.raw_files")
    a.out = a.out or C.get(cfg, "hdfs.processed_dir")
    sizes = C.get(cfg, "data.sample_sizes", {}) or {}
    a.sample_train = a.sample_train or sizes.get("train", 500000)
    a.sample_val = a.sample_val or sizes.get("val", 200000)
    a.sample_test = a.sample_test or sizes.get("test", 300000)
    a.scaler_out = a.scaler_out or C.get(cfg, "models.scaler", "models/scaler.json")
    a.executor_memory = a.executor_memory or C.get(cfg, "spark.executor_memory")
    a.driver_memory = a.driver_memory or C.get(cfg, "spark.driver_memory")
    a.shuffle_partitions = a.shuffle_partitions or C.get(cfg, "spark.shuffle_partitions", 12)
    a.persist_level = a.persist_level or C.get(cfg, "spark.persist_level", "MEMORY_AND_DISK")
    return a


@contextlib.contextmanager
def timed(stats: dict, name: str):
    t0 = time.time()
    yield
    stats["timings_s"][name] = round(time.time() - t0, 2)
    print("[time] %-16s %8.2fs" % (name, stats["timings_s"][name]), flush=True)


def col(name: str):
    """F.col an toàn với tên cột có dấu chấm/khoảng trắng."""
    from pyspark.sql import functions as F
    return F.col("`%s`" % name)


def build_spark(a):
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName("btl-preprocess") \
        .config("spark.sql.shuffle.partitions", str(a.shuffle_partitions)) \
        .config("spark.sql.ansi.enabled", "false")   # Spark 4 (Kaggle) bật ANSI mặc định; giữ hành vi cast của 3.5
    if a.master:
        b = b.master(a.master)
    if a.executor_memory:
        b = b.config("spark.executor.memory", a.executor_memory)
    if a.driver_memory:   # chỉ có tác dụng khi chạy bằng python (local); spark-submit dùng --driver-memory
        b = b.config("spark.driver.memory", a.driver_memory)
    return b.getOrCreate()


def read_raw(spark, paths):
    """Đọc từng file (mọi cột là string), bỏ cột định danh, gộp theo tên cột chung. Chưa bỏ header lặp."""
    frames, dropped = [], set()
    for p in paths:
        df = spark.read.option("header", True).csv(p)
        df = df.toDF(*[c.strip() for c in df.columns])
        ids = [c for c in ID_COLS if c in df.columns]
        dropped.update(ids)
        frames.append(df.drop(*ids))
    common = set.intersection(*[set(f.columns) for f in frames])
    cols = [c for c in frames[0].columns if c in common]
    for p, f in zip(paths, frames):
        extra = sorted(set(f.columns) - common)
        if extra:
            print("[warn] %s: bỏ cột không có ở mọi file: %s" % (p, extra))
    df = reduce(lambda x, y: x.unionByName(y), [f.select(*[col(c) for c in cols]) for f in frames])
    return df, sorted(dropped)


def drop_repeated_headers(df):
    return df.filter(col(LABEL_COL) != LABEL_COL)


def cast_and_clean(df):
    from pyspark.sql import functions as F
    feats = [c for c in df.columns if c not in (LABEL_COL, RAW_TS)]
    df = df.select(*[col(c).cast("double").alias(c) for c in feats],
                   F.trim(col(LABEL_COL)).alias(LABEL_COL), col(RAW_TS).alias(TS_COL))
    bad = None
    for c in feats:
        cond = col(c).isNull() | F.isnan(col(c)) | (F.abs(col(c)) == float("inf"))
        bad = cond if bad is None else (bad | cond)
    return df.filter(~bad), feats


def drop_constant(df, feats):
    from pyspark.sql import functions as F
    aggs = []
    for i, c in enumerate(feats):
        aggs += [F.min(col(c)).alias("mn%d" % i), F.max(col(c)).alias("mx%d" % i)]
    row = df.agg(*aggs).collect()[0]
    const = [c for i, c in enumerate(feats) if row["mn%d" % i] == row["mx%d" % i]]
    keep = [c for c in feats if c not in const]
    return df.drop(*const), keep, const


def dedup_with_ids(df, feats):
    """Gộp dòng trùng (feature + Label), giữ orig_ts nhỏ nhất; flow_id = xxhash64(feature..., Label)."""
    from pyspark.sql import functions as F
    keys = feats + [LABEL_COL]
    g = df.groupBy(*[col(c) for c in keys]).agg(F.min(col(TS_COL)).alias(TS_COL))
    return g.withColumn(ID_COL, F.xxhash64(*[col(c) for c in keys]))


def with_bucket(df):
    from pyspark.sql import functions as F
    return df.withColumn("_bucket", F.pmod(col(ID_COL), F.lit(100)))


def with_sample_key(df):
    """Khoá lấy mẫu tất định trong [0, 1), độc lập với bucket chia tập."""
    from pyspark.sql import functions as F
    return df.withColumn("_u", F.pmod(F.xxhash64(col(ID_COL), F.lit(SEED)), F.lit(1000000)) / 1e6)


def compute_scaler(train_df, feats) -> Scaler:
    from pyspark.sql import functions as F
    b = train_df.filter(col(LABEL_COL) == BENIGN)
    aggs = [F.avg(col(c)).alias("m%d" % i) for i, c in enumerate(feats)]
    aggs += [F.stddev_pop(col(c)).alias("s%d" % i) for i, c in enumerate(feats)]
    row = b.agg(*aggs).collect()[0]
    if row["m0"] is None:
        raise ValueError("train không có dòng Benign")
    mean = np.array([row["m%d" % i] for i in range(len(feats))], dtype=np.float64)
    std = np.array([row["s%d" % i] or 0.0 for i in range(len(feats))], dtype=np.float64)
    std[std == 0] = 1.0
    return Scaler(list(feats), mean, std)


def export_sample(df, n: int, total: int, path: str) -> None:
    """Lấy n dòng tất định (theo _u), ghi 1 file Parquet trong thư mục path."""
    from pyspark.sql import functions as F
    frac = min(1.0, 1.1 * n / max(total, 1))
    with_sample_key(df).filter(F.col("_u") < frac).orderBy("_u", ID_COL).limit(n).drop("_u") \
        .coalesce(1).write.mode("overwrite").parquet(path)


def demo_quota(counts: dict, n_rows: int, min_per_attack: int = DEMO_MIN_PER_ATTACK) -> dict:
    """Số dòng mỗi nhãn cho demo: mỗi loại tấn công >= min(min_per_attack, số có); phần còn lại chia theo tỷ lệ."""
    total = sum(counts.values())
    if total <= n_rows:
        return dict(counts)
    base = {k: (min(min_per_attack, v) if k != BENIGN else 0) for k, v in counts.items()}
    rem = max(n_rows - sum(base.values()), 0)
    avail = {k: counts[k] - base[k] for k in counts}
    tot_avail = sum(avail.values())
    return {k: base[k] + (int(rem * avail[k] / tot_avail) if tot_avail else 0) for k in counts}


def export_demo(test_df, counts: dict, n_rows: int, path: str) -> int:
    """demo_test.parquet: lấy mẫu phân tầng tất định, gom về driver (<= ~1.2*n_rows dòng) rồi ghi 1 file."""
    from pyspark.sql import functions as F
    quota = demo_quota(counts, n_rows)
    fracs = {k: min(1.0, 1.2 * q / max(counts[k], 1)) for k, q in quota.items()}
    cond = None
    for k, fr in fracs.items():
        c = (col(LABEL_COL) == k) & (F.col("_u") < fr)
        cond = c if cond is None else (cond | c)
    pdf = with_sample_key(test_df).filter(cond).toPandas()
    pdf = pdf.sort_values(["_u", ID_COL])
    pdf = pd.concat([g.head(quota[k]) for k, g in pdf.groupby(LABEL_COL, sort=True)])
    pdf = pdf.drop(columns=["_u"]).sort_values(ID_COL).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(path, index=False)
    return len(pdf)


def main(argv=None) -> dict:
    a = parse_args(argv)
    from pyspark import StorageLevel

    stats = {"timings_s": {}, "rows": {}, "files": list(a.files)}
    t_all = time.time()
    level = getattr(StorageLevel, a.persist_level)
    spark = build_spark(a)
    # --files nhận tên file (ghép với --input-dir) hoặc đường dẫn đầy đủ (/... hoặc hdfs://...)
    paths = [f if (f.startswith("/") or "://" in f) else "%s/%s" % (a.input_dir.rstrip("/"), f) for f in a.files]

    with timed(stats, "read_clean"):
        lines, id_dropped = read_raw(spark, paths)
        if a.debug_repartition:
            lines = lines.repartition(a.debug_repartition)
        stats["rows"]["raw_lines"] = lines.count()          # gồm cả các dòng header lặp
        raw = drop_repeated_headers(lines)
        stats["rows"]["raw"] = raw.count()
        n_cols_raw = len(raw.columns)
        clean, feats = cast_and_clean(raw)
        clean = clean.persist(level)
        stats["rows"]["after_clean"] = clean.count()
    with timed(stats, "constant_cols"):
        clean, feats, const = drop_constant(clean, feats)
        stats["n_features"] = len(feats)
        stats["features"] = feats
    with timed(stats, "dedup"):
        dedup = with_bucket(dedup_with_ids(clean, feats)).persist(level)
        stats["rows"]["after_dedup"] = dedup.count()
        clean.unpersist()
    r = stats["rows"]
    stats["dropped_columns"] = {"identifier": id_dropped + [RAW_TS + " (giữ thành orig_ts)"], "constant": const}
    stats["constant_cols_dropped"] = const
    stats["cleaning"] = [
        {"step": "read_csv", "rows": r["raw_lines"], "rows_removed": 0, "cols": n_cols_raw},
        {"step": "drop_repeated_headers", "rows": r["raw"], "rows_removed": r["raw_lines"] - r["raw"], "cols": n_cols_raw},
        {"step": "drop_nan_inf", "rows": r["after_clean"], "rows_removed": r["raw"] - r["after_clean"], "cols": n_cols_raw},
        {"step": "drop_constant_cols", "rows": r["after_clean"], "rows_removed": 0, "cols": n_cols_raw - len(const)},
        {"step": "drop_duplicates", "rows": r["after_dedup"], "rows_removed": r["after_clean"] - r["after_dedup"],
         "cols": n_cols_raw - len(const)},
    ]

    cols = [ID_COL, TS_COL] + feats + [LABEL_COL]
    split_dfs = {}
    for name, lo, hi in SPLITS:
        with timed(stats, "write_" + name):
            part = dedup.filter((col("_bucket") >= lo) & (col("_bucket") < hi)).select(*[col(c) for c in cols])
            part.write.mode("overwrite").parquet("%s/%s" % (a.out, name))
        split_dfs[name] = spark.read.parquet("%s/%s" % (a.out, name))
    dedup.unpersist()

    with timed(stats, "label_dist"):
        stats["label_distribution"] = {}
        for name, df in split_dfs.items():
            rows = df.groupBy(LABEL_COL).count().collect()
            stats["label_distribution"][name] = {row[LABEL_COL]: int(row["count"]) for row in rows}
            stats["rows"][name] = int(sum(row["count"] for row in rows))
    with timed(stats, "scaler"):
        compute_scaler(split_dfs["train"], feats).to_json(a.scaler_out)
    if a.export_sample:
        with timed(stats, "export_sample"):
            for name, n in (("train", a.sample_train), ("val", a.sample_val), ("test", a.sample_test)):
                export_sample(split_dfs[name], n, stats["rows"][name], "%s/%s" % (a.export_sample, name))
    if a.demo_out:
        with timed(stats, "export_demo"):
            stats["rows"]["demo"] = export_demo(split_dfs["test"], stats["label_distribution"]["test"],
                                                a.demo_rows, a.demo_out)

    stats["timings_s"]["total"] = round(time.time() - t_all, 2)
    Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps({"rows": stats["rows"], "label_distribution": stats["label_distribution"],
                      "timings_s": stats["timings_s"]}, indent=1, ensure_ascii=False), flush=True)
    return stats


if __name__ == "__main__":
    main()
