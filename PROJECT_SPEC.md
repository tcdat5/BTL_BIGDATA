# PROJECT SPEC — Real-time Network Anomaly Detection (BTL_BigDeep)

> File này mô tả toàn bộ đề tài cho coding agent. Đọc hết trước khi viết code.
> Đặt file ở gốc repo (có thể copy thành `AGENTS.md` / `CLAUDE.md` tuỳ agent).

## 1. Mục tiêu

Xây dựng hệ thống **phát hiện bất thường mạng thời gian thực** dùng **Deep Autoencoder (AE)** trên nền **Apache Kafka + Apache Spark**, dữ liệu **CSE-CIC-IDS2018** (network flow CSV).

Sản phẩm cuối phải có đủ:
1. Hệ thống Big Data chạy được: dữ liệu trên HDFS, xử lý bằng Spark, truyền qua Kafka, chạy trên cụm 3 VM.
2. So sánh **4 model Deep Learning** (Deep AE — model chính, VAE, Denoising AE, DNN/MLP classifier) với **6 baseline ML** trên **cùng tập test, cùng quy trình đánh giá**, kèm **thí nghiệm ablation** cho AE (đây là môn Deep Learning nên phần so sánh kiến trúc DL là bắt buộc).
3. Kết quả thực nghiệm: Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, throughput, latency (p50/p95/p99) + biểu đồ.
4. Dashboard demo: số flow đã xử lý, số cảnh báo, reconstruction error, các flow đáng ngờ.
5. (Ngoài phạm vi code) Báo cáo khoa học — code phải xuất đủ bảng/hình để đưa vào báo cáo.

**Deadline: sáng thứ 6 (16/10/2026).** Ưu tiên chạy được end-to-end trước, tối ưu sau. Không thêm tính năng ngoài spec.

## 2. Môi trường

| Thành phần | Chi tiết |
|---|---|
| Cụm | 3 VM VirtualBox: `master` + `slave1` + `slave2` |
| Trên cụm | HDFS, Spark (standalone, master trên `master`), Kafka (đặt trên `master`) |
| Train model | **Kaggle Notebook** (GPU cho các model DL, CPU cho baseline ML) |
| Suy luận trên cụm | `onnxruntime` (CPU) — **không cài PyTorch lên VM** |
| Ngôn ngữ | Python 3, PySpark |

**Chưa biết, KHÔNG được đoán cứng trong code:** hostname/IP, port, phiên bản Spark/Kafka, RAM/CPU mỗi VM, đường dẫn HDFS.
→ Mọi giá trị này để trong `config.yaml` (có `config.example.yaml` commit vào repo). Hỏi người dùng nếu cần.

**Đã xác nhận (12/10/2026):** Spark 3.5.1 (Scala 2.13), Java 8 (OpenJDK 1.8.0_402), Hadoop 3.3.0, Python 3.8.10 trên cụm. Kafka 3.7.0 (Scala 2.13), KRaft, single broker `master:9092`. Connector: `org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1`. ONNX opset <= 17. Smoke test local dùng Java 8 hoặc 11.

## 3. Cấu trúc repo

```
.
├── PROJECT_SPEC.md
├── README.md                 # cách cài + chạy từng bước, lệnh copy-paste được
├── config.example.yaml
├── requirements-cluster.txt  # pyspark, kafka-python, onnxruntime, pandas, pyarrow, numpy, streamlit, pyyaml
├── requirements-kaggle.txt   # torch, onnx, scikit-learn, lightgbm, pandas, pyarrow, matplotlib
├── common/
│   ├── data.py               # load_split, scaler (dùng CHUNG cho AE và baseline)
│   └── metrics.py            # evaluate() — dùng CHUNG
├── preprocess/
│   └── spark_preprocess.py   # chạy trên cụm, CSV trên HDFS -> Parquet train/val/test + scaler.json
├── train/
│   ├── baselines.py          # ĐÃ CÓ — refactor để import từ common/, giữ nguyên hành vi
│   ├── dl_models.py          # định nghĩa AE, VAE, DAE, MLP (PyTorch)
│   ├── train_dl.py           # train 1 model DL theo --model, đánh giá, export ONNX
│   ├── run_ablation.py       # chạy lưới ablation cho AE, gộp kết quả
│   └── kaggle_notebook.ipynb # hoặc hướng dẫn chạy 2 script trên Kaggle
├── analysis/
│   ├── make_report_assets.py # gộp metric, vẽ biểu đồ -> reports/
│   └── eda.py                # trực quan hoá dữ liệu -> reports/eda/
├── streaming/
│   ├── producer.py           # replay test set vào Kafka với tốc độ chỉnh được
│   ├── stream_detect.py      # Spark Structured Streaming + ONNX
│   └── benchmark.py          # đo throughput/latency từ kết quả streaming
├── dashboard/
│   └── app.py                # Streamlit
├── scripts/
│   ├── check_cluster.sh      # kiểm tra HDFS/Spark/Kafka sống
│   └── run_*.sh              # lệnh spark-submit, v.v.
├── models/                   # ae.onnx, scaler.json, threshold.json (gitignore file lớn)
└── results/  reports/        # gitignore output lớn, giữ CSV tổng hợp nhỏ
```

## 4. Quy ước dữ liệu (QUAN TRỌNG — mọi module phải tuân theo)

- Cột nhãn: `Label`. Giá trị bình thường: `Benign`. Mọi giá trị khác là tấn công (vd `DoS attacks-Hulk`, `Bot`, `DDOS attack-HOIC`, `FTP-BruteForce`, `SSH-Bruteforce`...).
- Nhị phân hoá: `y = 1` nếu `Label != "Benign"`.
- **`models/scaler.json`** là nguồn duy nhất cho danh sách feature và chuẩn hoá:
  ```json
  {"features": ["Flow Duration", "..."], "mean": [...], "std": [...]}
  ```
  - Fit **chỉ trên Benign của tập train**. `std == 0` → thay bằng 1.
  - Transform: `clip((x - mean) / std, -10, 10)` dạng float32. Hàm này phải giống hệt nhau ở baseline, AE và streaming.
- **`models/threshold.json`**: `{"percentile": 99, "threshold": <float>}` — ngưỡng của AE = percentile 99 reconstruction error trên **val Benign**. Không chỉnh ngưỡng bằng tập test.
- Tập test để đánh giá: lấy mẫu `--test-cap` (mặc định 300000) với `random_state=42` qua **cùng một hàm** `common.data.load_split` → AE và baseline đánh giá trên đúng cùng các dòng.

## 5. Các thành phần

### 5.1 `preprocess/spark_preprocess.py` (chạy trên cụm)
Input: các CSV CSE-CIC-IDS2018 trên HDFS (mặc định 4 ngày: 14-02, 16-02, 21-02, 02-03-2018; danh sách lấy từ config).
Các bước:
1. Đọc CSV (header). Loại các dòng header lặp lại giữa file (dòng có `Label == "Label"`).
2. Chuẩn hoá schema giữa các file (file 20-02 có thêm `Flow ID`, `Src IP`, `Dst IP`, `Src Port` — bỏ đi nếu có).
3. Bỏ cột định danh/gây leakage: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Timestamp` (giữ `Timestamp` riêng thành cột metadata `orig_ts` cho streaming, KHÔNG dùng làm feature), `Dst Port` (bỏ khỏi feature).
4. Cast các cột feature sang double; `Infinity`/`-Infinity`/`NaN` → xử lý nhất quán (thay bằng null rồi drop dòng, hoặc thay 0 — chọn một và ghi trong README).
5. Bỏ cột hằng (variance = 0 trên toàn bộ dữ liệu).
6. `dropDuplicates` trên feature + Label **trước** khi chia tập.
7. Chia train/val/test = 70/10/20, stratified theo `Label` (dùng `sampleBy` hoặc random split theo từng Label), seed 42.
8. Ghi Parquet: `<out>/train`, `<out>/val`, `<out>/test`. Ghi thêm bản **đã lấy mẫu nhỏ** để upload Kaggle (`--export-sample`, vd train 500k / val 200k / test 300k) ra thư mục cục bộ.
9. Tính `scaler.json` từ Benign trong train.
10. In ra: số dòng mỗi tập, phân phối Label mỗi tập, thời gian chạy (để đưa vào báo cáo).

Phải chạy được cả `--master spark://...` (cụm) và `--master local[*]` (fallback trên Kaggle).

### 5.2 `train/baselines.py` (ĐÃ CÓ)
Đã viết và đã test: PCA reconstruction, Isolation Forest, One-Class SVM (mẫu con 20k) — nhóm unsupervised; Random Forest, Logistic Regression, LightGBM — nhóm supervised. Có `--holdout-attack` cho thí nghiệm tấn công chưa thấy.
Việc cần làm: chuyển `load_split`, scaler, `evaluate` sang `common/`, đọc `scaler.json` từ `models/`. **Không đổi định dạng output.**

Output hiện tại (giữ nguyên, AE phải xuất cùng định dạng):
- `results/baselines_metrics.csv`: cột `model, group, precision, recall, f1, fpr, roc_auc, pr_auc, threshold, train_time_s, infer_ms_per_1k`
- `results/baselines_per_attack.csv`: `model, attack, recall, n`
- `results/scores_<model>.npz`: `y`, `score`, `thr`

### 5.3 Các model Deep Learning (chạy trên Kaggle GPU)

**`train/dl_models.py`** — định nghĩa 4 model PyTorch, tất cả dùng MLP vì dữ liệu là dạng bảng (mỗi flow là một bản ghi độc lập):

| `--model` | Tên trong bảng kết quả | Nhóm | Train trên | Điểm bất thường (score) |
|---|---|---|---|---|
| `ae` | `DeepAutoencoder` | unsupervised | Benign | MSE tái tạo theo mẫu |
| `vae` | `VariationalAE` | unsupervised | Benign | MSE tái tạo dùng **μ** (không lấy mẫu) lúc suy luận |
| `dae` | `DenoisingAE` | unsupervised | Benign | MSE tái tạo trên đầu vào **sạch** |
| `mlp` | `DNN_Classifier` | supervised | Train đầy đủ có nhãn | Xác suất lớp tấn công (sigmoid) |

Chi tiết kiến trúc:
- **AE**: encoder `d → 64 → 32 → latent(16)`, decoder đối xứng; ReLU, BatchNorm sau mỗi Linear ẩn; tầng ra tuyến tính. `--hidden 64,32 --latent 16` là tham số CLI.
- **VAE**: encoder giống AE nhưng ra `μ` và `logσ²` (cùng kích thước latent), reparameterization trick, decoder giống AE. Loss = MSE (tổng theo feature, trung bình theo batch) + `β`·KL, `--beta` mặc định 1.0, có **KL warm-up** tuyến tính trong 5 epoch đầu để tránh posterior collapse. Ghi riêng recon_loss và kl_loss vào history.
- **DAE**: kiến trúc giống hệt AE; lúc train cộng nhiễu Gaussian `N(0, σ²)` vào đầu vào (`--noise-std` mặc định 0.1), loss so với đầu vào sạch. Lúc suy luận không cộng nhiễu.
- **MLP**: `d → 128 → 64 → 32 → 1`, ReLU + BatchNorm + Dropout 0.2, `BCEWithLogitsLoss` với `pos_weight = n_benign / n_attack` của tập train. Ngưỡng 0.5 (giống nhóm supervised ML). Hỗ trợ `--holdout-attack` như `baselines.py`.

**`train/train_dl.py --model {ae,vae,dae,mlp}`** — một script chung cho cả 4:
- Dữ liệu và scaler qua `common/data.py` (giống baseline, cùng tập test lấy mẫu).
- Adam lr 1e-3, batch 1024, tối đa 50 epoch, early stopping theo loss trên val (unsupervised: val Benign; MLP: toàn bộ val) với patience 5, giữ trọng số tốt nhất. Nạp toàn bộ tensor lên GPU nếu đủ bộ nhớ. Seed 42 (torch, numpy, cudnn deterministic).
- Ngưỡng unsupervised = percentile `--percentile` (mặc định 99) của score trên **val Benign**. Ghi `models/threshold_<model>.json`.
- Đánh giá test qua `common.metrics.evaluate` → `results/dl_<model>_metrics.csv`, `results/dl_<model>_per_attack.csv`, `results/scores_<TênModel>.npz` (cùng định dạng với baseline).
- Lưu `results/dl_<model>_history.csv` (epoch, train_loss, val_loss, và với VAE thêm recon/kl).
- Ghi số tham số của model và thời gian train vào file metric (thêm cột `n_params`).
- Export `models/<model>.onnx`: input `input` `[batch, d]` float32, batch dynamic; output `score` — **đưa luôn phép tính score vào graph ONNX** (với AE/VAE/DAE: graph trả về MSE theo mẫu; với MLP: trả về xác suất). Như vậy streaming chỉ cần so `score >= threshold` cho mọi model. Kiểm tra bằng `onnxruntime`: sai khác so với PyTorch < 1e-4 trên 1000 mẫu.
- Copy model AE thành `models/ae.onnx` + `models/threshold.json` cho streaming (mặc định streaming dùng AE; đổi được qua config).

**`train/run_ablation.py`** — thí nghiệm ablation cho **AE** (mỗi lần chạy giảm tối đa còn 20 epoch để tiết kiệm thời gian, ghi rõ trong output):
1. Kích thước latent: 4, 8, 16, 32.
2. Độ sâu: `hidden=32` (nông), `hidden=64,32` (mặc định), `hidden=128,64,32` (sâu).
3. Ngưỡng: percentile 95, 97, 99, 99.5, 99.9 — **không train lại**, chỉ tính lại trên score của model mặc định.
4. (Tuỳ thời gian) VAE với β = 0.5, 1, 2.

Output: `results/ablation.csv` với cột `experiment, value, precision, recall, f1, fpr, roc_auc, pr_auc, n_params, train_time_s`.

**Không làm** LSTM/CNN/Transformer: mỗi flow là bản ghi độc lập và cột IP đã bị bỏ để tránh leakage, nên không có chuỗi hợp lệ để mô hình hoá. Ghi vào mục "hướng phát triển" của báo cáo.

### 5.4 `analysis/make_report_assets.py`
- Gộp `baselines_metrics.csv` + mọi `dl_*_metrics.csv` → `reports/model_comparison.csv` và `.md` (bảng Markdown, 4 chữ số thập phân). Thêm cột `family` (`DL` / `ML`), sắp theo group rồi F1. In đậm DeepAutoencoder.
- Bảng recall theo loại tấn công (model × attack) → `reports/per_attack_recall.csv` + heatmap.
- Hình PNG (matplotlib, dpi 150):
  - ROC và PR chung, **tách 2 hình**: nhóm unsupervised (AE, VAE, DAE, PCA, IF, OCSVM) và nhóm supervised (MLP, RF, LR, LightGBM).
  - Histogram score Benign vs Attack (trục log, vạch ngưỡng) cho AE, VAE, DAE.
  - Đường loss train/val của cả 4 model DL; riêng VAE vẽ thêm recon và KL.
  - Bar F1 + FPR theo model, tô màu theo DL/ML.
  - Ablation: F1 và FPR theo latent size; theo độ sâu (kèm số tham số); đường Precision/Recall/FPR theo percentile ngưỡng.
- Nếu có kết quả holdout → bảng recall của loại bị holdout: AE/VAE/DAE so với MLP/RF/LightGBM.

### 5.5 `streaming/producer.py`
- Đọc Parquet test (cục bộ hoặc HDFS), gửi từng flow vào Kafka topic (mặc định `network-flows`) dạng JSON.
- Message schema:
  ```json
  {"flow_id": 123, "produce_ts": 1739500000123, "orig_ts": "14/02/2018 08:31:01",
   "label": "Benign", "features": {"Flow Duration": 1234.0, "...": 0.0}}
  ```
  `produce_ts` = epoch milliseconds lúc gửi. `label` chỉ dùng để đánh giá, **model không được dùng**.
- `--rate` (bản ghi/giây, có điều tiết), `--limit`, `--loop`. Dùng gửi bất đồng bộ + `linger_ms`/batch để đạt được rate cao. In rate thực tế mỗi 5 giây.

### 5.6 `streaming/stream_detect.py` (Spark Structured Streaming)
- Đọc Kafka topic, parse JSON theo danh sách feature trong `scaler.json`.
- Suy luận trong `foreachBatch` (ưu tiên, dễ debug) hoặc `pandas_udf`: chuẩn hoá giống mục 4 → `onnxruntime` chạy `ae.onnx` (load một lần mỗi executor/process, không load mỗi batch). Output `score` của ONNX chính là reconstruction error → cột `recon_error`; `is_anomaly = recon_error >= threshold` (ngưỡng đọc từ `threshold.json`). Đường dẫn model/ngưỡng lấy từ config để có thể đổi sang model khác.
- Ghi kết quả ra Parquet (thư mục cấu hình được, cục bộ hoặc HDFS) với cột: `flow_id, orig_ts, label, recon_error, is_anomaly, produce_ts, kafka_ts, detect_ts` (`kafka_ts` = timestamp của Kafka record; `detect_ts` = epoch ms lúc ghi).
- Ghi thêm `results/stream_batches.csv`: `batch_id, n_rows, batch_start_ts, batch_end_ts, duration_ms`.
- Tham số: `--trigger` (giây), `--max-offsets-per-trigger`, `--starting-offsets`.
- Có checkpoint location.

### 5.7 `streaming/benchmark.py`
- Đọc output streaming của một lần chạy. Tính: throughput (số dòng / (max detect_ts − min detect_ts)), latency end-to-end = `detect_ts − produce_ts` (p50/p95/p99/max), latency Kafka→detect.
- Tính Precision/Recall/F1/FPR của streaming (dùng cột `label`) để chứng minh khớp kết quả offline.
- Append một dòng vào `results/streaming_benchmark.csv` với các tham số chạy (`rate, trigger, n_workers, ...`) truyền vào qua CLI.
- Hình: throughput theo rate, latency p50/p95/p99 theo rate, so sánh 1 vs 2 worker.

### 5.8 `dashboard/app.py` (Streamlit)
- Đọc thư mục output streaming, tự refresh (vd 3 giây).
- Hiển thị: tổng flow đã xử lý, số cảnh báo, tỷ lệ cảnh báo, throughput gần nhất; biểu đồ số flow/cảnh báo theo thời gian; biểu đồ reconstruction error kèm đường ngưỡng; bảng các flow đáng ngờ gần nhất (recon_error cao nhất).
- Phân loại theo nhãn thật chỉ hiển thị ở tab "Evaluation" riêng (vì hệ thống thật không có nhãn).

### 5.9 `scripts/check_cluster.sh`
Kiểm tra: `hdfs dfsadmin -report` (số datanode sống), Spark master UI phản hồi + số worker, tạo/gửi/nhận thử message Kafka. In PASS/FAIL từng mục.

### 5.10 `analysis/eda.py` — trực quan hoá dữ liệu (chạy trên Kaggle CPU hoặc laptop)
Input: Parquet mẫu (`data/kaggle_sample/` = Kaggle Dataset), `models/scaler.json`, `results/preprocess_stats.json`. Đọc qua `common/data.py`. Output: `reports/eda/*.png` (dpi 150) + `reports/eda/eda_summary.md`. Mọi phép chiếu 2D dùng **mẫu phân tầng** (tối đa 2.000 dòng mỗi loại nhãn, `--per-class`), seed 42.

**Phần 1 — chạy ngay sau preprocessing (M1):**
1. **Phân phối nhãn**: bar trục log số dòng theo từng loại tấn công, tách train/val/test; kèm tỷ lệ Benign/Attack. Lấy từ `preprocess_stats.json` (đếm trên toàn bộ dữ liệu, không phải mẫu).
2. **Hiệu quả làm sạch**: bảng số dòng trước/sau từng bước (header lặp, NaN/Inf, trùng lặp, cột hằng bị bỏ) từ `preprocess_stats.json`. Nếu preprocessing chưa ghi đủ các số này thì bổ sung vào `spark_preprocess.py`.
3. **Feature phân biệt nhất**: tính KS statistic Benign vs Attack cho mọi feature → top 12; vẽ violin/box trên thang `log1p` (Benign vs từng loại tấn công). Lưu bảng KS đầy đủ ra CSV.
4. **Tương quan**: heatmap Pearson trên Benign train (sắp theo hierarchical clustering); liệt kê các cặp |r| > 0.95. Dùng để lập luận dữ liệu dư thừa → hợp với nén bằng AE.
5. **Sau chuẩn hoá**: histogram vài feature trước/sau `transform`; tỷ lệ giá trị bị clip ở ±10 cho từng feature.
6. **Chiếu 2D dữ liệu gốc** (đã chuẩn hoá), tô màu theo loại nhãn: PCA (kèm biểu đồ explained variance tích luỹ), t-SNE (perplexity 30), UMAP nếu cài được `umap-learn` (bỏ qua nếu không có, không làm lỗi).

**Phần 2 — sau khi train DL (M2a), cờ `--latent`:**
7. **Không gian latent**: lấy vector latent của AE (encoder output) và VAE (μ) trên **cùng mẫu** ở mục 6 → chiếu PCA/t-SNE 2D, đặt cạnh hình PCA dữ liệu gốc (figure 1×3). Đây là hình minh hoạ chính cho phần Deep Learning: biểu diễn học được tách Benign/Attack thế nào so với dữ liệu gốc.
   - Yêu cầu `train_dl.py` lưu thêm `models/<kind>.pt` (state_dict + tham số kiến trúc) để `eda.py` dựng lại encoder. ONNX không đổi.
8. Tính thêm **silhouette score** theo nhãn nhị phân trên dữ liệu gốc vs latent AE vs latent VAE → bảng nhỏ trong `eda_summary.md` (định lượng cho hình mục 7).

`eda_summary.md`: liệt kê mọi hình kèm 1 câu mô tả *cái hình thể hiện* (không diễn giải kết quả bằng số chưa tính).
Thời gian chạy mục tiêu: < 10 phút trên Kaggle CPU (t-SNE là phần chậm nhất; giảm `--per-class` nếu cần).

## 6. Thứ tự thực hiện (milestone)

| # | Hạn | Việc | Tiêu chí xong |
|---|---|---|---|
| M0 | Thứ 2 tối | Khung repo, `config.example.yaml`, `common/`, `check_cluster.sh`, README | Repo chạy `pip install` được, script check chạy |
| M1 | Thứ 3 trưa | `spark_preprocess.py` | Có Parquet train/val/test + `scaler.json` + bản mẫu để upload Kaggle |
| M2a | Thứ 3 chiều | `dl_models.py`, `train_dl.py` (làm AE trước, rồi VAE/DAE/MLP), refactor `baselines.py` | Có `ae.onnx`, `threshold.json`; 4 model DL + 6 baseline chạy xong trên Kaggle |
| M1b | Thứ 3 chiều | `analysis/eda.py` phần 1 (mục 1–6) — chạy trên Kaggle CPU song song lúc GPU train | Có `reports/eda/` với hình phân phối nhãn, KS, tương quan, PCA/t-SNE |
| M2b | Thứ 3 tối | `run_ablation.py`, holdout, `make_report_assets.py`, `eda.py --latent` (mục 7–8) | Có `ablation.csv`, bảng so sánh 10 model, hình latent space + toàn bộ hình |
| M3 | Thứ 4 chiều | `producer.py`, `stream_detect.py` | Chạy end-to-end trên cụm, F1 streaming ≈ F1 offline của AE |
| M4 | Thứ 4 tối | `dashboard/app.py` | Dashboard hiện cảnh báo khi producer đang chạy |
| M5 | Thứ 5 sáng | `benchmark.py` + script chạy các mức rate | Có `streaming_benchmark.csv` + hình |

Làm theo đúng thứ tự. Nếu bị chặn ở bước cần cụm thật, viết code + test local trước (`local[*]`, Kafka chạy bằng Docker nếu có), rồi báo người dùng lệnh để chạy trên cụm.

## 7. Quy tắc cho coding agent

- **Không bịa kết quả.** Mọi con số trong README/report phải sinh ra từ code đã chạy. Chưa chạy thì để trống.
- Mỗi script có `argparse`, `--help` rõ ràng, đọc `config.yaml`, không hard-code IP/đường dẫn.
- Mỗi module phải có cách **chạy thử nhanh trên dữ liệu giả nhỏ** (vd `--smoke` hoặc `tests/` sinh Parquet giả vài nghìn dòng) — dùng nó để tự kiểm tra trước khi giao.
- Dùng chung `common/` — không chép lại logic chuẩn hoá ở nơi khác.
- Seed cố định 42 ở mọi chỗ có ngẫu nhiên.
- Log thời gian chạy của mọi bước nặng (để đưa vào báo cáo).
- VM có RAM hạn chế: Spark job phải cấu hình được `executor-memory`, không `collect()`/`toPandas()` toàn bộ dữ liệu lớn.
- Không cài PyTorch trên cụm; cụm chỉ cần `onnxruntime`.
- Ưu tiên khi thiếu thời gian: AE > MLP > VAE > ablation latent/ngưỡng > DAE > ablation độ sâu > VAE β. Streaming chỉ cần **một** model (AE).
- Có `--smoke` cho `train_dl.py` (vài nghìn dòng giả, 2 epoch, chạy được trên CPU) để kiểm tra cả 4 model + export ONNX trước khi đưa lên Kaggle.
- Commit nhỏ, theo từng milestone. Cập nhật README sau mỗi milestone với lệnh chạy chính xác.
- Khi phải chọn giữa đúng-và-đơn-giản với tối ưu, chọn đúng-và-đơn-giản.
