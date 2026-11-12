#!/usr/bin/env bash
# Benchmark nhiều mức rate. Mỗi run: topic mới -> stream_detect nền -> warm-up -> producer đo -> chờ detector dừng
# -> benchmark -> xoá topic/checkpoint (mỗi message ~2.5 KB, giữ lại sẽ đầy đĩa master).
# Dùng: RATES="1000 2000 4000 8000" WORKERS=2 DURATION=60 bash scripts/run_benchmark.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
RATES="${RATES:-1000 2000 4000 8000}"
WORKERS="${WORKERS:-2}"
DURATION="${DURATION:-60}"
WARMUP="${WARMUP:-2000}"                 # số flow warm-up (gửi 500 msg/s), không tính vào kết quả
TRIGGER="${TRIGGER:-$(cfg streaming.trigger_s)}"
MAXOFF="${MAXOFF:-$(cfg streaming.max_offsets_per_trigger)}"
INFER="${INFER:-$(cfg streaming.inference)}"
INPUT="${INPUT:-$(cfg hdfs.processed_dir)/test}"
KH=$(cfg cluster.kafka_home)
BS=$(cfg cluster.kafka_bootstrap)
export TOTAL_CORES=$((WORKERS * $(cfg spark.executor_cores)))
mkdir -p results/stream logs

for RATE in $RATES; do
  TAG="r${RATE}_w${WORKERS}_$(date +%s)"
  OUT="results/stream/$TAG"
  TOPIC="bench-$TAG"
  LIMIT=$((RATE * DURATION))
  echo "=== $TAG: rate=$RATE limit=$LIMIT cores=$TOTAL_CORES"
  "$KH/bin/kafka-topics.sh" --bootstrap-server "$BS" --create --if-not-exists --topic "$TOPIC" \
    --partitions "$(cfg kafka.partitions)" --replication-factor 1 >/dev/null
  bash scripts/run_stream.sh --topic "$TOPIC" --starting-offsets earliest --out-dir "$OUT" \
    --checkpoint "checkpoints/$TAG" --batches-csv "$OUT/_batches.csv" --trigger "$TRIGGER" \
    --max-offsets-per-trigger "$MAXOFF" --inference "$INFER" --stop-when-idle 20 \
    --timeout $((DURATION * 4 + 240)) --clean > "logs/$TAG.stream.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 180); do [ -f "$OUT/_STARTED" ] && break; sleep 1; done
  [ -f "$OUT/_STARTED" ] || { echo "stream_detect không khởi động, xem logs/$TAG.stream.log"; kill $SPID; exit 1; }

  # warm-up: executor khởi động Python worker + nạp ONNX ở batch đầu tiên
  python3 streaming/producer.py --topic "$TOPIC" --input "$INPUT" --rate 500 --limit "$WARMUP" >/dev/null
  for _ in $(seq 1 120); do ls "$OUT"/batch-*.parquet >/dev/null 2>&1 && break; sleep 1; done
  sleep $(( $(printf '%.0f' "$TRIGGER") * 3 ))          # xả hết flow warm-up còn lại
  date +%s%3N > "$OUT/_MEASURE_START"

  python3 streaming/producer.py --topic "$TOPIC" --input "$INPUT" --rate "$RATE" --limit "$LIMIT" \
    --stats-out "$OUT/_producer.json"
  wait $SPID || true
  python3 streaming/benchmark.py --out-dir "$OUT" --rate "$RATE" --trigger "$TRIGGER" --n-workers "$WORKERS" \
    --max-offsets "$MAXOFF" --inference "$INFER" --tag "$TAG" --producer-stats "$OUT/_producer.json" \
    || echo "benchmark lỗi cho $TAG (xem logs/$TAG.stream.log)"
  "$KH/bin/kafka-topics.sh" --bootstrap-server "$BS" --delete --topic "$TOPIC" >/dev/null 2>&1 || true
  rm -rf "checkpoints/$TAG"
done
echo "Kết quả: results/streaming_benchmark.csv"
# master không có matplotlib (không có Internet) -> vẽ trên laptop: python streaming/benchmark.py --plot
python3 streaming/benchmark.py --plot 2>/dev/null && echo "Hình: reports/stream_*.png" \
  || echo "Không vẽ được hình trên máy này; chép CSV về laptop rồi chạy: python streaming/benchmark.py --plot"
