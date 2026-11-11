# BTL_BigDeep — Real-time Network Anomaly Detection (CSE-CIC-IDS2018)

Deep Autoencoder + Kafka + Spark Structured Streaming trên cụm 3 VM. Spec đầy đủ: `PROJECT_SPEC.md`.
Kế hoạch: `docs/superpowers/plans/2026-10-12-network-anomaly-detection.md`.

## Môi trường
| Nơi | Python | Ghi chú |
|---|---|---|
| Cụm (master, slave1, slave2) | 3.8.10 | Spark 3.5.1 (Scala 2.13), Java 8, Hadoop 3.3.0, Kafka 3.7.0 KRaft trên master |
| Kaggle | 3.11+ | GPU cho DL, CPU cho baseline |
| Dev local | 3.12 venv (phần train) + Docker (phần cụm: Python 3.8 + Java 8 + Kafka 3.7.0) | |

`requirements-cluster.txt` đã kiểm chứng cài được trên Python 3.8 (image `docker/Dockerfile.cluster-test`):
pyspark 3.5.1, kafka-python 2.0.6, onnxruntime 1.17.3, numpy 1.24.4, pandas 2.0.3, pyarrow 14.0.2,
pyyaml 6.0.1, streamlit 1.39.0, matplotlib 3.7.5.

## Cài đặt
```bash
cp config.example.yaml config.yaml        # sửa hostname/port/đường dẫn HDFS/RAM theo cụm thật
python3 -m pip install --user -r requirements-cluster.txt   # trên CẢ 3 VM
```
Đọc một giá trị config trong shell: `python3 scripts/cfg.py cluster.kafka_bootstrap`.

## Tiền xử lý
- Giá trị lỗi: dòng có null/NaN/±Infinity ở **bất kỳ** feature nào bị **bỏ**.
- Bỏ cột định danh `Flow ID, Src IP, Dst IP, Src Port, Dst Port`; `Timestamp` giữ thành metadata `orig_ts`; bỏ cột hằng.
- Gộp dòng trùng (feature + Label), giữ `orig_ts` nhỏ nhất.
- Chia tập **tất định bằng hash** (giống hệt nhau trên Kaggle và trên cụm, không phụ thuộc số partition):
  `flow_id = xxhash64(feature..., Label)`, `bucket = pmod(flow_id, 100)`: `<70` train, `70–79` val, `≥80` test.
- Xuất thêm: bản mẫu Kaggle (`data.sample_sizes`), `data/demo_test.parquet` (50.000 dòng test phân tầng, mỗi
  loại tấn công ≥ min(500, số có)), `models/scaler.json`, `results/preprocess_stats.json`.
```bash
# Kaggle / local
python preprocess/spark_preprocess.py --master "local[*]" --input-dir <thư mục CSV> --out data/processed \
  --export-sample data/kaggle_sample --demo-out data/demo_test.parquet --scaler-out models/scaler.json
# Cụm (sau này)
bash scripts/run_preprocess.sh
```

## Chạy trên Kaggle (Deep Learning, toàn bộ pipeline)
Notebook: `train/kaggle_notebook.ipynb` (sinh từ `train/make_notebook.py`), logic trong `train/kaggle_run_all.py`.
1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → chọn `train/kaggle_notebook.ipynb`.
2. **Add Data**: dataset CSV CSE-CIC-IDS2018 có 4 ngày 14-02, 16-02, 21-02, 02-03-2018. Sửa `RAW_DIR` trong ô tham số
   thành thư mục `/kaggle/input/<tên dataset>` (notebook tìm CSV đệ quy; tên file khác config được tự khớp theo ngày,
   hoặc đặt `FILE_MAP`).
3. **Settings → Accelerator: GPU T4 x2** (hoặc P100); **Internet: On**.
4. Lấy mã nguồn — một trong hai cách:
   - repo public: để `REPO_URL` đúng, `CODE_DATASET = ""` (notebook `git clone`);
   - repo private: trên laptop `git archive -o btl_code.zip HEAD`, upload thành Kaggle Dataset (vd `btl-bigdeep-code`),
     Add Data vào notebook, đặt `CODE_DATASET = "/kaggle/input/btl-bigdeep-code"`.
5. Đặt `QUICK = True` → **Save Version → Save & Run All** (thử < 15 phút). Ổn thì `QUICK = False` và chạy lại.
6. Tab **Output** của version → tải `kaggle_outputs.zip`.

Bước lỗi: in traceback rồi chạy tiếp; preprocess hoặc AE lỗi thì dừng. Bảng thời gian: `results/kaggle_timing.csv`.
Thử cục bộ (Docker giống Kaggle: Python 3.12 + Java 17):
```bash
docker compose -f docker/docker-compose.test.yml --profile kaggle build kaggle
docker compose -f docker/docker-compose.test.yml --profile kaggle run --rm kaggle sh -c \
  "python -m common.fake --raw-csv .cache/quick/input --n 3000 && python train/kaggle_run_all.py --raw-dir .cache/quick/input --work-dir .cache/quick/working --quick"
```

## Dashboard trên laptop (replay, không cần Kafka/Spark/PyTorch)
```powershell
# trong thư mục repo: giải nén output Kaggle -> models/ results/ reports/ data/demo_test.parquet
Expand-Archive -Force kaggle_outputs.zip .
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py                     # mặc định --source replay
```
Sidebar: chọn model (mọi `models/*.onnx`), cách đặt ngưỡng (đã lưu / percentile val Benign / tự chỉnh), tốc độ phát,
Start/Pause/Reset, "Phát nhanh 1.000 flow". Tab: Giám sát (không nhãn) · Đánh giá (nhãn thật) · Kết quả nghiên cứu.
Nguồn Spark Streaming (sau này): `streamlit run dashboard/app.py -- --source stream --stream-dir results/stream/default`.

## Cụm: dữ liệu, kiểm tra, streaming (chạy trên master)
**Dữ liệu.** Tải 4 CSV (giữ tên dài như `data.raw_files`) vào `data/raw/` rồi đưa lên HDFS. Không cần Internet trên VM:
tải trên laptop rồi share/scp sang.
```bash
aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" data/raw/
# tương tự Friday-16-02-2018, Wednesday-21-02-2018, Friday-02-03-2018
hdfs dfs -mkdir -p /user/$USER/cicids2018/raw && hdfs dfs -put data/raw/*.csv /user/$USER/cicids2018/raw/
```
Repo để trong VirtualBox Shared Folder thì chỉ đọc code/CSV từ đó; `streaming.output_dir`, `checkpoint_dir` để trên
đĩa local của VM hoặc HDFS (vboxsf chậm và lỗi rename/lock). Gọi script bằng `bash scripts/...` (vboxsf mất quyền +x).

**Kiểm tra cụm.**
```bash
bash scripts/check_cluster.sh            # đọc config.yaml; in [PASS]/[FAIL] cho HDFS, Spark, Kafka
```

**Streaming.** Cần `models/ae.onnx`, `models/threshold.json`, `models/scaler.json` (từ `kaggle_outputs.zip` hoặc
`run_preprocess.sh`). Producer gửi **feature gốc** (chưa chuẩn hoá); `stream_detect` chuẩn hoá bằng `scaler.json`.
Inference mặc định chạy **trên executor** (`foreachBatch` + `mapInPandas`): cả 3 VM phải cài `requirements-cluster.txt`;
model/scaler/threshold được gửi qua `SparkContext.addFile`. Nếu slave thiếu package: thêm `--inference driver`.
Output: `results/stream/default/batch-*.parquet` (đĩa master) + `results/stream_batches.csv`.
```bash
# Terminal 1: detector
bash scripts/run_stream.sh --clean --starting-offsets latest
# Terminal 2: producer — test đầy đủ (hdfs) cho benchmark, hoặc data/demo_test.parquet cho demo
python3 streaming/producer.py --create-topic --input hdfs:///user/$USER/cicids2018/processed/test --rate 2000 --limit 200000
python3 streaming/producer.py --input data/demo_test.parquet --rate 500
# Terminal 3: dashboard
streamlit run dashboard/app.py -- --source stream --stream-dir results/stream/default
# Đổi model: --model models/vae.onnx --threshold models/threshold_vae.json
```

## Test local
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-kaggle.txt
.venv\Scripts\python -m pytest -q                                                      # phần train + common
docker compose -f docker/docker-compose.test.yml build cluster
docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest -q  # phần cụm (Py3.8 + Java 8)
docker compose -f docker/docker-compose.test.yml run --rm cluster pytest -q            # kèm Kafka 3.7.0 (test kafka/e2e)
```
Máy dev Windows có Smart App Control: numpy 2.5 bị chặn DLL `numpy.random._mt19937` → trong venv dùng
`pip install "numpy==2.2.6"` (chỉ cho máy dev, Kaggle không cần).
