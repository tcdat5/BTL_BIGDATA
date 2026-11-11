#!/usr/bin/env bash
# Chạy stream_detect trên cụm. Tham số thêm chuyển thẳng cho streaming/stream_detect.py.
# Số core executor tổng: TOTAL_CORES (mặc định spark.total_executor_cores) — dùng để so 1 vs 2 worker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
# Connector Kafka: VM không có Internet -> dùng jar đã có trong ~/.ivy2/jars (hoặc KAFKA_JARS=a.jar,b.jar);
# không đủ jar thì mới để spark-submit tải qua --packages.
if [ -z "${KAFKA_JARS:-}" ]; then
  KAFKA_JARS=$(ls "$HOME"/.ivy2/jars/*spark-sql-kafka-0-10_2.13-3.5.1.jar "$HOME"/.ivy2/jars/*spark-token-provider-kafka-0-10_2.13-3.5.1.jar \
    "$HOME"/.ivy2/jars/*kafka-clients-*.jar "$HOME"/.ivy2/jars/*commons-pool2-*.jar 2>/dev/null | paste -sd, -)
  [ "$(printf '%s' "$KAFKA_JARS" | tr ',' '\n' | grep -c .)" -eq 4 ] || KAFKA_JARS=""
fi
if [ -n "$KAFKA_JARS" ]; then DEPS=(--jars "$KAFKA_JARS"); else DEPS=(--packages "$(cfg cluster.kafka_package)"); fi
echo "[run_stream] ${DEPS[*]}"
spark-submit --master "$(cfg cluster.spark_master)" \
  "${DEPS[@]}" \
  --driver-memory "$(cfg spark.driver_memory)" --executor-memory "$(cfg spark.executor_memory)" \
  --executor-cores "$(cfg spark.executor_cores)" \
  --total-executor-cores "${TOTAL_CORES:-$(cfg spark.total_executor_cores)}" \
  --conf spark.pyspark.python=python3 \
  streaming/stream_detect.py "$@"
