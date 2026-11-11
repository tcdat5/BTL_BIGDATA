#!/usr/bin/env bash
# Kiểm tra HDFS / Spark / Kafka. Dùng: bash scripts/check_cluster.sh [config.yaml]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BTL_CONFIG="${1:-${BTL_CONFIG:-$ROOT/config.yaml}}"
cfg() { python3 "$ROOT/scripts/cfg.py" "$1"; }
FAIL=0
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }

# 1. HDFS
EXPECTED_DN=$(cfg cluster.expected_datanodes)
LIVE=$(hdfs dfsadmin -report 2>/dev/null | sed -n 's/^Live datanodes (\([0-9]*\)).*/\1/p' | head -1)
if [ -n "$LIVE" ] && [ "$LIVE" -ge "$EXPECTED_DN" ]; then
  pass "HDFS: $LIVE datanode sống"
else
  fail "HDFS: live datanodes='${LIVE:-?}', cần >= $EXPECTED_DN"
fi

# 2. Spark master UI
UI=$(cfg cluster.spark_master_ui)
EXPECTED_W=$(cfg cluster.expected_workers)
if command -v curl >/dev/null 2>&1; then
  JSON=$(curl -s --max-time 5 "$UI/json/" || true)
else   # VM không có curl và không có Internet để cài
  JSON=$(python3 -c 'import sys, urllib.request as u; print(u.urlopen(sys.argv[1] + "/json/", timeout=5).read().decode())' "$UI" 2>/dev/null || true)
fi
if [ -n "$JSON" ]; then
  W=$(printf '%s' "$JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for w in d.get("workers", []) if w.get("state") == "ALIVE"))' 2>/dev/null || echo 0)
  if [ "$W" -ge "$EXPECTED_W" ]; then pass "Spark: master UI OK, $W worker ALIVE"; else fail "Spark: $W worker ALIVE, cần >= $EXPECTED_W"; fi
else
  fail "Spark: master UI $UI không phản hồi"
fi

# 3. Kafka: tạo topic, gửi, nhận
KH=$(cfg cluster.kafka_home)
BS=$(cfg cluster.kafka_bootstrap)
T="btl-healthcheck"
MSG="ping-$(date +%s)-$$"
"$KH/bin/kafka-topics.sh" --bootstrap-server "$BS" --create --if-not-exists --topic "$T" \
  --partitions 1 --replication-factor 1 >/dev/null 2>&1
echo "$MSG" | "$KH/bin/kafka-console-producer.sh" --bootstrap-server "$BS" --topic "$T" >/dev/null 2>&1
GOT=$("$KH/bin/kafka-console-consumer.sh" --bootstrap-server "$BS" --topic "$T" --from-beginning \
  --timeout-ms 10000 2>/dev/null | grep -c "$MSG")
if [ "${GOT:-0}" -ge 1 ]; then pass "Kafka: gửi/nhận message OK ($BS)"; else fail "Kafka: không nhận lại được message trên $BS"; fi

exit $FAIL
