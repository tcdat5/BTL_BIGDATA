# Mô tả bài tập lớn Big Data: Phát hiện bất thường mạng thời gian thực

## 1. Tên đề tài

**Nghiên cứu và xây dựng hệ thống phát hiện bất thường mạng thời gian thực sử dụng Deep Autoencoder trên nền tảng Apache Kafka và Apache Spark.**

*Tiếng Anh: Real-time Network Anomaly Detection System Using Deep Autoencoder with Apache Kafka and Apache Spark.*

## 2. Bài toán

Lưu lượng mạng thường lẫn các hành vi đáng ngờ như DoS/DDoS, brute force hoặc botnet. Luật phát hiện cố định chỉ bắt được kiểu tấn công đã biết. Lưu lượng phát sinh liên tục với khối lượng lớn, nên hệ thống phải vừa xử lý dữ liệu lớn vừa đưa ra cảnh báo trong thời gian ngắn.

Đề tài giải quyết hai vấn đề:

- **Dữ liệu lớn:** tiếp nhận, lưu trữ và xử lý phân tán lượng lớn lưu lượng mạng bằng HDFS, Spark và Kafka.
- **Phát hiện bất thường:** học đặc điểm của lưu lượng bình thường và phát hiện các flow khác biệt. Mô hình chính là Deep Autoencoder, chỉ học từ lưu lượng Benign, nên về lý thuyết có thể bắt được tấn công chưa từng thấy.

## 3. Dữ liệu

- **Bộ dữ liệu:** CSE-CIC-IDS2018, file CSV các network flow do CICFlowMeter trích xuất.
- **Phạm vi:** 4 ngày 14-02, 16-02, 21-02 và 02-03-2018. Các loại tấn công có trong phạm vi: FTP-BruteForce, SSH-Bruteforce (14-02); DoS-Hulk, DoS-SlowHTTPTest (16-02); DDoS-LOIC-UDP, DDoS-HOIC (21-02); Bot (02-03).
- **Nguồn tải:** AWS S3 công khai `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/` (`aws s3 cp --no-sign-request`), hoặc Kaggle Dataset `solarmainframe/ids-intrusion-csv` (tên file dạng `02-14-2018.csv`, notebook tự khớp theo ngày).
- **Đặc điểm "bẩn" cần xử lý:**
  - Dòng header lặp lại **ngay trong file** (ví dụ file 16-02).
  - Cột `Flow Byts/s`, `Flow Pkts/s` có giá trị Infinity hoặc NaN.
  - Một số file của bộ dữ liệu (ví dụ 20-02) có thêm các cột `Flow ID`, `Src IP`, `Dst IP`, `Src Port`. Code tự bỏ các cột định danh nếu file nào có, nên vẫn chạy được khi mở rộng sang các ngày khác.
- **Quy ước nhãn:** cột `Label`. `Benign` là bình thường, các giá trị khác là tấn công. Nhị phân hoá thành `y = 1` nếu không phải Benign.

## 4. Kiến trúc hệ thống

```
CSV (CSE-CIC-IDS2018)
   │
   ▼
HDFS  ──►  Spark (preprocess)  ──►  Parquet train/val/test + scaler.json
                                          │
                                          ▼
                              Kaggle GPU: train model  ──►  *.onnx + threshold.json
                                                                 │
Python Producer ──► Kafka (topic network-flows) ──► Spark Structured Streaming
   (replay test set,                              (chuẩn hoá bằng scaler.json,
    feature gốc, có produce_ts)                    onnxruntime chấm điểm,
                                                   so score ≥ threshold)
                                                                 │
                                                    Parquet kết quả + batches.csv
                                                                 │
                                          ┌──────────────────────┴───────────┐
                                    Streamlit dashboard                 benchmark.py
                                    (giám sát, đánh giá)          (throughput, latency, F1)
```

**Nguyên tắc nối các thành phần:** cụm và Kaggle chỉ trao đổi qua file. Kaggle nhận Parquet và `scaler.json` từ cụm, cụm nhận lại `*.onnx` và `threshold.json` từ Kaggle. Mọi graph ONNX trả thẳng điểm bất thường, nên streaming chỉ cần so `score >= threshold`, không cần biết kiến trúc model.

**Môi trường:** cụm 3 VM VirtualBox (`master`, `slave1`, `slave2`) do giảng viên cung cấp, chạy trên một máy vật lý.

| Thành phần | Phiên bản |
|---|---|
| Spark | 3.5.1 (Scala 2.13) |
| Java | 8 (OpenJDK 1.8.0_402) |
| Hadoop | 3.3.0 |
| Python | 3.8.10 |
| Kafka | 3.7.0 (Scala 2.13), chế độ KRaft, một broker tại `master:9092` |

## 5. Các giai đoạn

### Giai đoạn 0: Chuẩn bị dữ liệu và cụm

1. Tải 4 file CSV, giữ tên dài như trong `data.raw_files` của config.
2. Đưa lên HDFS: `hdfs dfs -put ... /user/<user>/cicids2018/raw/`.
3. Cài `requirements-cluster.txt` trên cả 3 VM, đặt `PYSPARK_PYTHON=python3`.
4. Cấu hình Kafka KRaft, tạo topic `network-flows` (3 partition).
5. Tạo `config.yaml` từ `config.example.yaml`, chạy `bash scripts/check_cluster.sh`: HDFS, Spark và Kafka đều phải `[PASS]`.

### Giai đoạn 1: Tiền xử lý trên Spark (chạy trên cụm)

Job `spark_preprocess.py` thực hiện:

1. Đọc CSV, bỏ các dòng header lặp.
2. Chuẩn hoá schema giữa các file.
3. Bỏ cột định danh và cột gây leakage: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Dst Port`. Cột `Timestamp` được giữ thành `orig_ts` làm metadata, không dùng làm feature.
4. Ép kiểu double. Dòng có null, NaN hoặc ±Infinity ở bất kỳ feature nào bị bỏ.
5. Bỏ cột hằng.
6. Bỏ trùng lặp trên feature và Label **trước** khi chia tập.
7. **Chia tập tất định theo hash:** `bucket = pmod(xxhash64(features, Label), 100)`. Bucket dưới 70 là train, 70–79 là val, từ 80 là test. Cách này cho kết quả giống hệt nhau dù số partition khác nhau, và vì đã bỏ trùng trước khi chia nên không có flow nào vừa nằm trong train vừa nằm trong test.
8. Fit scaler chỉ trên Benign của train, lưu `scaler.json`.
9. Ghi Parquet, kèm bản mẫu nhỏ cho Kaggle, `demo_test.parquet` (50.000 dòng test phân tầng) và `preprocess_stats.json` để vẽ EDA.

### Giai đoạn 2: Huấn luyện và so sánh mô hình (chạy trên Kaggle)

**Model chính: Deep Autoencoder.** Encoder `d → 64 → 32 → 16`, decoder đối xứng, dùng ReLU và BatchNorm. Train chỉ trên Benign, loss MSE. Điểm bất thường là reconstruction error theo từng flow. Ngưỡng là percentile 99 của error trên tập val Benign. Không chỉnh ngưỡng bằng tập test.

**Các model so sánh:**

| Nhóm | Model | Ghi chú |
|---|---|---|
| Deep Learning | AE (chính), VAE, Denoising AE, MLP | MLP là mạng có giám sát, dùng nhãn |
| Baseline ML không giám sát | PCA reconstruction, Isolation Forest, One-Class SVM | OCSVM train trên mẫu 20k |
| Baseline ML có giám sát | Random Forest, Logistic Regression, LightGBM | Dùng nhãn |

**Thí nghiệm bổ sung:**

- **EDA:** phân phối nhãn, hiệu quả làm sạch, top feature theo KS, tương quan, chiếu PCA/t-SNE dữ liệu gốc. Sau khi train thêm latent space của AE/VAE và silhouette score.
- **Ablation AE:** kích thước latent (4/8/16/32), độ sâu mạng (3 mức), percentile ngưỡng (95 đến 99.9).
- **Holdout:** loại một kiểu tấn công khỏi tập train của nhóm có giám sát (Bot, DDOS attack-HOIC, SSH-Bruteforce), để kiểm tra khả năng phát hiện tấn công chưa thấy.
- **Tổng hợp:** bảng so sánh 10 model, ROC/PR, phân phối score, đường loss (`make_report_assets.py`).

**Thiết kế đánh giá:** mọi model dùng cùng tập test (lấy mẫu 300k với seed 42), cùng scaler, cùng hàm metric. Metric gồm Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, và recall theo từng loại tấn công.

Kết thúc giai đoạn: tải `kaggle_outputs.zip`, chép `models/ae.onnx`, `models/threshold.json` và `models/scaler.json` vào `models/` trên master.

### Giai đoạn 3: Pipeline thời gian thực (chạy trên cụm)

- **Producer (Python, kafka-python):** đọc tập test (test đầy đủ trên HDFS cho benchmark, `demo_test.parquet` cho demo), gửi từng flow vào Kafka dạng JSON với **feature gốc, chưa chuẩn hoá**. Mỗi message có `flow_id`, `produce_ts` (epoch ms, để đo latency), `orig_ts`, `label`, `features`. Tốc độ phát chỉnh được qua `--rate`. Nhãn thật chỉ đi kèm để đánh giá, model không dùng.
- **Spark Structured Streaming:** đọc topic, parse JSON theo danh sách feature trong `scaler.json`, chuẩn hoá, chấm điểm bằng ONNX Runtime trong `foreachBatch` + `mapInPandas`, chạy trên executor. Model và scaler phân phối qua `SparkContext.addFile`. Nếu slave không cài được onnxruntime thì dùng `--inference driver`. Kết quả ghi ra Parquet với các cột `flow_id, orig_ts, label, recon_error, is_anomaly, produce_ts, kafka_ts, detect_ts`. Mỗi micro-batch ghi một dòng vào `stream_batches.csv`. Có checkpoint.
- **Tham số:** trigger interval, `maxOffsetsPerTrigger`, starting offsets, số worker.

### Giai đoạn 4: Dashboard và đánh giá

- **Dashboard Streamlit** có hai nguồn:
  - `stream`: đọc thư mục output streaming, tự làm mới mỗi 3 giây.
  - `replay`: tự chấm `demo_test.parquet` bằng ONNX, không cần Kafka hay Spark. Dùng làm phương án demo dự phòng.
  - Hiển thị tổng flow, số cảnh báo, tỷ lệ cảnh báo, throughput, biểu đồ theo thời gian, biểu đồ score kèm đường ngưỡng, và bảng các flow đáng ngờ. Nhãn thật chỉ hiện ở tab Đánh giá riêng.
- **Benchmark:** tính throughput, latency end-to-end (`detect_ts − produce_ts`) với p50/p95/p99, và F1 của streaming để so với F1 offline. Mục đích là chứng minh pipeline không làm sai kết quả. Đo ở các mức rate 1.000 đến 20.000 bản ghi/giây, với 1 và 2 worker. Kết quả ghi vào `results/streaming_benchmark.csv` và vẽ hình.

## 6. Mục tiêu đo lường

- **Chất lượng phát hiện:** F1, Precision, Recall, FPR, ROC-AUC, PR-AUC.
- **Hiệu năng:** throughput (mục tiêu ban đầu ít nhất 1.000 bản ghi/giây, cần đo thực tế), latency p50/p95/p99.
- **Khả năng mở rộng:** so sánh 1 và 2 worker. Hạn chế là hai worker chạy trên cùng một máy vật lý nên không phải mở rộng thật.

## 7. Trạng thái hiện tại

| Phần | Trạng thái |
|---|---|
| Khung repo, config, môi trường test Docker (Python 3.8, Java 8, Spark 3.5.1) | Xong, có test |
| `common/` (data, metrics, fake data) | Xong |
| Preprocess Spark, chia tập theo hash | Xong, đã kiểm tra chia tập giống nhau khi đổi số partition |
| Baseline ML (`baselines.py`) | Xong, kết quả khớp bản gốc trong sai số 1e-4 |
| Deep Learning (AE, VAE, DAE, MLP) và export ONNX | Xong ở mức code và smoke test, ONNX sai khác PyTorch tối đa 3.8e-6 |
| Notebook Kaggle | Đã chạy QUICK trên dữ liệu giả, chưa chạy trên dữ liệu thật |
| Dashboard replay | Xong, đã kiểm tra với output giả |
| `check_cluster.sh`, `producer.py`, `stream_detect.py`, `run_stream.sh` | Đã viết, test pass trong Docker (Python 3.8 + Java 8 + Spark local). Chưa chạy trên cụm thật và chưa chạy test e2e với Kafka |
| Kafka 3.7.0 trên master | Đã cài, đang cấu hình KRaft và kiểm tra kết nối |
| `benchmark.py`, `run_benchmark.sh` | Chưa làm |
| `eda.py`, `run_ablation.py`, `make_report_assets.py` | Chưa làm |

Phần Big Data đã hoãn từ thứ 3 để ưu tiên môn Deep Learning, nên phần streaming trên cụm chưa có kết quả nào. Khi nộp, nên ghi rõ phần đã làm và phần còn dở trong báo cáo. Không ghi số liệu streaming khi chưa đo.

## 8. Hạn chế cần nêu trong báo cáo

- Dữ liệu là lịch sử, hệ thống mô phỏng thời gian thực bằng cách phát lại. Timestamp gốc chỉ là metadata.
- Ba VM chạy trên cùng một máy vật lý, tài nguyên tranh chấp nhau.
- Chỉ dùng 4 ngày dữ liệu, không phải toàn bộ bộ CSE-CIC-IDS2018. Các loại tấn công khó như Infiltration (28-02, 01-03) và tấn công Web (22-02, 23-02) nằm ngoài phạm vi; đây là hướng mở rộng.
- Mô hình xử lý từng flow độc lập, không khai thác chuỗi thời gian giữa các flow.
- Ngưỡng dựa trên percentile của val Benign, không tối ưu cho từng loại tấn công.

## 9. Sản phẩm nộp

1. Hệ thống chạy được: HDFS, Spark, Kafka, dashboard.
2. Bảng so sánh 10 model và thí nghiệm ablation.
3. Bảng metric và benchmark: Precision, Recall, F1, FPR, throughput, latency.
4. Biểu đồ: EDA, ROC/PR, phân phối score, đường loss, latent space, throughput theo rate.
5. Báo cáo khoa học: nghiên cứu liên quan, phương pháp, kiến trúc, thực nghiệm, so sánh, hạn chế, hướng phát triển.
