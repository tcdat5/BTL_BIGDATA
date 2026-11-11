#!/usr/bin/env bash
# Chạy tiền xử lý trên cụm rồi kéo bản mẫu Kaggle + scaler + stats về master.
# Dùng: bash scripts/run_preprocess.sh [tham số thêm cho spark_preprocess.py]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
SAMPLE_HDFS=$(cfg hdfs.sample_dir)
SAMPLE_LOCAL=$(cfg data.sample_local_dir)
mkdir -p results data

spark-submit --master "$(cfg cluster.spark_master)" \
  --driver-memory "$(cfg spark.driver_memory)" --executor-memory "$(cfg spark.executor_memory)" \
  --executor-cores "$(cfg spark.executor_cores)" \
  --total-executor-cores "${TOTAL_CORES:-$(cfg spark.total_executor_cores)}" \
  --conf spark.pyspark.python=python3 \
  preprocess/spark_preprocess.py --export-sample "$SAMPLE_HDFS" --demo-out data/demo_test.parquet "$@" \
  2>&1 | tee results/preprocess.log

rm -rf "$SAMPLE_LOCAL" && mkdir -p "$SAMPLE_LOCAL"
for s in train val test; do hdfs dfs -get "$SAMPLE_HDFS/$s" "$SAMPLE_LOCAL/$s"; done
cp "$(cfg models.scaler)" "$SAMPLE_LOCAL/scaler.json"
cp results/preprocess_stats.json "$SAMPLE_LOCAL/preprocess_stats.json"
echo "Bản mẫu Kaggle: $SAMPLE_LOCAL ; demo: data/demo_test.parquet"
