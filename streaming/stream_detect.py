"""Spark Structured Streaming: Kafka -> chuẩn hoá -> ONNX -> Parquet (PROJECT_SPEC 5.6).

Inference (--inference):
  executor (mặc định): foreachBatch + mapInPandas chạy trên executor. Cả 3 VM cần requirements-cluster.txt;
                       model/scaler/threshold gửi bằng SparkContext.addFile, code (common/, streaming/) bằng addPyFile;
                       mỗi Python worker load ONNX một lần (streaming.scorer.get_scorer).
  driver:              foreachBatch + toPandas() micro-batch, suy luận trên driver (chỉ master cần onnxruntime).
Kết quả mỗi micro-batch (đã giới hạn bởi --max-offsets-per-trigger) gom về driver và ghi
<out-dir>/batch-<id>.parquet trên đĩa của máy chạy driver (master); dashboard/benchmark đọc thư mục này.
Cụm:  bash scripts/run_stream.sh [tham số]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import load_scaler  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS  # noqa: E402

OUT_SCHEMA = ("flow_id long, orig_ts string, label string, produce_ts long, kafka_ts long, "
              "recon_error double, is_anomaly boolean")
BATCH_COLUMNS = ["batch_id", "n_rows", "batch_start_ts", "batch_end_ts", "duration_ms"]


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--master", default=None, help="vd local[2]; bỏ trống khi dùng spark-submit --master")
    p.add_argument("--bootstrap", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--model", default=None, help="mặc định models.stream_model")
    p.add_argument("--threshold", default=None, help="mặc định models.stream_threshold")
    p.add_argument("--scaler", default=None, help="mặc định models.scaler")
    p.add_argument("--out-dir", default=None, help="thư mục cục bộ trên driver")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batches-csv", default=None)
    p.add_argument("--trigger", type=float, default=None, help="giây")
    p.add_argument("--max-offsets-per-trigger", type=int, default=None)
    p.add_argument("--starting-offsets", default=None, choices=["latest", "earliest"])
    p.add_argument("--inference", default=None, choices=["executor", "driver"])
    p.add_argument("--kafka-package", default=None, help="chỉ dùng khi chạy bằng python (spark-submit dùng --packages)")
    p.add_argument("--timeout", type=float, default=0, help="dừng sau N giây (0 = không)")
    p.add_argument("--stop-when-idle", type=float, default=0, help="dừng khi N giây không có dữ liệu mới (sau khi đã có)")
    p.add_argument("--clean", action="store_true", help="xoá out-dir + checkpoint cục bộ trước khi chạy")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    defaults = {"bootstrap": "cluster.kafka_bootstrap", "topic": "kafka.topic", "model": "models.stream_model",
                "threshold": "models.stream_threshold", "scaler": "models.scaler", "out_dir": "streaming.output_dir",
                "checkpoint": "streaming.checkpoint_dir", "batches_csv": "streaming.batches_csv",
                "trigger": "streaming.trigger_s", "max_offsets_per_trigger": "streaming.max_offsets_per_trigger",
                "starting_offsets": "streaming.starting_offsets", "inference": "streaming.inference"}
    for attr, key in defaults.items():
        if getattr(a, attr) is None:
            setattr(a, attr, C.get(cfg, key))
    if a.kafka_package is None and a.master:     # spark-submit đã nạp connector qua --jars/--packages
        a.kafka_package = C.get(cfg, "cluster.kafka_package")
    return a


def build_spark(a):
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName("btl-stream-detect") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "4")
    if a.master:
        b = b.master(a.master)
    if a.kafka_package:
        b = b.config("spark.jars.packages", a.kafka_package)
    return b.getOrCreate()


def prepare_executor_assets(spark, a) -> None:
    """Gửi model/scaler/threshold + code cho executor."""
    sc = spark.sparkContext
    for f in (a.model, a.scaler, a.threshold):
        sc.addFile(str(Path(f).resolve()))
    zpath = Path(tempfile.gettempdir()) / "btl_bigdeep_code.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for pkg in ("common", "streaming"):
            for f in sorted((ROOT / pkg).rglob("*.py")):
                z.write(f, f.relative_to(ROOT).as_posix())
    sc.addPyFile(str(zpath))


def parse_stream(raw, features):
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType
    feat = StructType([StructField(f, DoubleType()) for f in features])
    schema = StructType([StructField("flow_id", LongType()), StructField("produce_ts", LongType()),
                         StructField("orig_ts", StringType()), StructField("label", StringType()),
                         StructField("features", feat)])
    m = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("m"),
                   (F.col("timestamp").cast("double") * 1000).cast("long").alias("kafka_ts"))
    cols = [F.col("m.flow_id").alias("flow_id"), F.col("m.orig_ts").alias("orig_ts"),
            F.col("m.label").alias("label"), F.col("m.produce_ts").alias("produce_ts"), F.col("kafka_ts")]
    cols += [F.col("m.features").getField(f).alias("f%d" % i) for i, f in enumerate(features)]
    return m.select(*cols).filter(F.col("flow_id").isNotNull() & F.col("produce_ts").isNotNull())


def _make_infer(names, feat_cols):
    model_name, scaler_name, thr_name = names

    def infer(it):
        from pyspark import SparkFiles

        from streaming.scorer import get_scorer, score_frame
        scorer = get_scorer(SparkFiles.get(model_name), SparkFiles.get(scaler_name), SparkFiles.get(thr_name))
        for pdf in it:
            yield score_frame(pdf, scorer, feat_cols)
    return infer


def append_batch_row(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


class BatchHandler:
    def __init__(self, a, n_features: int):
        self.a = a
        self.feat_cols = ["f%d" % i for i in range(n_features)]
        self.out = Path(a.out_dir)
        self.names = (Path(a.model).name, Path(a.scaler).name, Path(a.threshold).name)
        self.rows_seen = 0
        self.last_data_ms = None

    def __call__(self, batch_df, batch_id):
        t0 = now_ms()
        if self.a.inference == "executor":
            pdf = batch_df.mapInPandas(_make_infer(self.names, self.feat_cols), OUT_SCHEMA).toPandas()
        else:
            from streaming.scorer import get_scorer, score_frame
            scorer = get_scorer(self.a.model, self.a.scaler, self.a.threshold)
            pdf = score_frame(batch_df.toPandas(), scorer, self.feat_cols)
        n = len(pdf)
        if n:
            pdf["detect_ts"] = now_ms()
            dst = self.out / ("batch-%08d.parquet" % batch_id)
            tmp = self.out / ("batch-%08d.parquet.tmp" % batch_id)
            pdf[OUTPUT_COLUMNS].to_parquet(tmp, index=False)
            os.replace(tmp, dst)
            self.rows_seen += n
            self.last_data_ms = now_ms()
        t1 = now_ms()
        append_batch_row(self.a.batches_csv, {"batch_id": batch_id, "n_rows": n, "batch_start_ts": t0,
                                              "batch_end_ts": t1, "duration_ms": t1 - t0})
        print("[stream] batch %d: %d rows in %d ms (total %d)" % (batch_id, n, t1 - t0, self.rows_seen), flush=True)


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.clean:
        for p in (a.out_dir, a.checkpoint):
            if not str(p).startswith("hdfs://"):
                shutil.rmtree(p, ignore_errors=True)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    features = load_scaler(a.scaler).features
    spark = build_spark(a)
    if a.inference == "executor":
        prepare_executor_assets(spark, a)
    reader = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", a.bootstrap) \
        .option("subscribe", a.topic) \
        .option("startingOffsets", a.starting_offsets) \
        .option("failOnDataLoss", "false")
    if a.max_offsets_per_trigger:
        reader = reader.option("maxOffsetsPerTrigger", str(a.max_offsets_per_trigger))
    handler = BatchHandler(a, len(features))
    q = parse_stream(reader.load(), features).writeStream \
        .foreachBatch(handler) \
        .option("checkpointLocation", a.checkpoint) \
        .trigger(processingTime="%g seconds" % a.trigger) \
        .start()
    (Path(a.out_dir) / "_STARTED").write_text(str(now_ms()), encoding="utf-8")
    print("[stream] query started: topic=%s inference=%s out=%s" % (a.topic, a.inference, a.out_dir), flush=True)
    t_start = time.time()
    try:
        while q.isActive:
            q.awaitTermination(1)
            if a.timeout and time.time() - t_start > a.timeout:
                print("[stream] timeout %gs -> stop" % a.timeout, flush=True)
                break
            if a.stop_when_idle and handler.last_data_ms and now_ms() - handler.last_data_ms > a.stop_when_idle * 1000:
                print("[stream] idle %gs -> stop" % a.stop_when_idle, flush=True)
                break
    except KeyboardInterrupt:
        print("[stream] dừng theo yêu cầu", flush=True)
    finally:
        q.stop()
    return {"rows": handler.rows_seen}


if __name__ == "__main__":
    main()
