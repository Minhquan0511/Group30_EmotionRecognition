# Nhóm 30 - Nhận diện cảm xúc khuôn mặt trên FER-2013

Đây là dự án môn Nhập môn học máy, xây dựng pipeline nhận diện cảm xúc từ ảnh
khuôn mặt và so sánh ba mô hình có độ phức tạp tăng dần:

1. `softmax`: hồi quy Softmax trên 2.304 giá trị pixel.
2. `mlp`: mạng truyền thẳng với một lớp ẩn gồm 128 nút.
3. `cnn`: CNN nhỏ gồm ba khối tích chập.

Ba mô hình dùng cùng cách chia dữ liệu, class weight và các chỉ số Accuracy,
Macro-F1 để việc so sánh công bằng. Pipeline hiện tại hỗ trợ CLAHE, chuẩn hóa
theo mean và độ lệch chuẩn của tập train, cùng tăng cường dữ liệu cho tập train.
Mã nguồn hiện tại chưa huấn luyện ResNet18; đây là hướng phát triển cho báo cáo
cuối kỳ và demo webcam.

## 1. Cài đặt

Khuyến nghị sử dụng Python 3.10-3.12:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

### Cách A - Tệp CSV gốc của FER-2013

Tải `fer2013.csv` từ cuộc thi FER-2013 trên Kaggle và đặt tại:

```text
data/fer2013.csv
```

Sau đó chuyển dữ liệu sang tệp NumPy:

```bash
python prepare_data.py --csv data/fer2013.csv --output data/fer2013.npz
```

Chương trình giữ nguyên cách chia trong cột `Usage` của dữ liệu gốc:

- `Training`: dùng để huấn luyện;
- `PublicTest`: dùng làm validation và chọn checkpoint;
- `PrivateTest`: chỉ dùng để đánh giá cuối cùng.

### Cách B - Bộ ảnh đã giải nén

Nếu dữ liệu có cấu trúc `train/<class>` và `test/<class>`, chạy:

```bash
python prepare_image_folders.py --root /đường-dẫn/đến-bộ-dữ-liệu \
  --output data/fer2013_full.npz
```

Tên thư mục tải về trên máy không ảnh hưởng đến chương trình. Script kiểm tra
bắt buộc đủ 28.709 ảnh train, 3.589 ảnh validation và 3.589 ảnh test.

### Tập con để chạy thử nhanh

Lệnh sau tải một tập con từ bản FER-2013 Enhanced công khai trên Hugging Face.
Cách này chỉ phù hợp để kiểm tra chương trình, không thay thế thí nghiệm trên
toàn bộ FER-2013 trong báo cáo cuối kỳ.

```bash
python fetch_subset.py --train-size 2100 --val-size 700 --test-size 700 \
  --output data/fer2013_subset.npz
```

## 3. Tiền xử lý và tăng cường dữ liệu

Pipeline đầy đủ thực hiện các bước sau:

- đưa ảnh về ảnh xám kích thước 48 × 48 pixel;
- áp dụng CLAHE với clip limit mặc định là 2,0 và lưới 8 × 8;
- tính mean và độ lệch chuẩn chỉ trên tập train, sau đó dùng hai giá trị này để
  chuẩn hóa train, validation và test;
- chỉ tăng cường dữ liệu trên tập train bằng lật ngang với xác suất 50%, xoay
  từ -10° đến 10°, co giãn trong khoảng 0,90-1,10, dịch tối đa 4 pixel và thay
  đổi nhẹ độ sáng, tương phản;
- không augmentation tập validation và test.

Cách làm này giúp tăng độ đa dạng của dữ liệu huấn luyện mà không làm thay đổi
dữ liệu dùng để lựa chọn và đánh giá mô hình.

## 4. Chạy thí nghiệm

Chạy cả ba mô hình trên toàn bộ dữ liệu với pipeline tiền xử lý đầy đủ:

```bash
python train_compare.py --data data/fer2013_full.npz \
  --models softmax mlp cnn --epochs 20 --batch-size 64 \
  --clahe --standardize --augment \
  --output-dir results/full_preprocessing_run
```

Ý nghĩa của các tùy chọn:

- `--clahe`: tăng tương phản cục bộ của ảnh;
- `--standardize`: chuẩn hóa bằng mean và độ lệch chuẩn chỉ tính từ tập train;
- `--augment`: áp dụng các biến đổi ngẫu nhiên cho ảnh train;
- `--models`: chọn một hoặc nhiều mô hình cần huấn luyện;
- `--output-dir`: thư mục lưu kết quả của lần chạy.

Kết quả của lần chạy 20 epoch trên toàn bộ FER-2013:

| Mô hình | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Thời gian train |
|---|---:|---:|---:|---:|
| Softmax | 22,68% | 24,85% | 22,26% | 26,39 giây |
| MLP | 30,12% | 36,53% | 30,67% | 30,45 giây |
| CNN nhỏ | 45,40% | 52,08% | 46,43% | 367,35 giây |

CNN nhỏ cho kết quả tốt nhất vì có thể khai thác quan hệ không gian giữa các
vùng trên khuôn mặt. Softmax và MLP được dùng làm mô hình đối chứng để cho thấy
ảnh hưởng của cấu trúc tích chập, không phải là mô hình cuối cùng cho demo.

Chạy nhanh trên tập con:

```bash
python train_compare.py --data data/fer2013_subset.npz \
  --models softmax mlp cnn --epochs 5 --batch-size 64 \
  --output-dir results/progress_run
```

Sau mỗi lần chạy, chương trình tạo:

- `summary.csv` và `summary.json`: Accuracy, Macro-F1 và thời gian huấn luyện;
- `*_confusion_matrix.png`: ma trận nhầm lẫn trên tập test;
- `*_history.png`: loss và accuracy theo epoch;
- `*_best.pt`: checkpoint tốt nhất theo validation Macro-F1;
- `run_config.json`: cấu hình và thông tin tiền xử lý để tái lập thí nghiệm.

## 5. Demo webcam thời gian thực

Sau khi có checkpoint CNN, kết nối webcam và chạy:

```bash
python realtime_demo.py \
  --checkpoint results/full_preprocessing_run/cnn_best.pt \
  --camera 0
```

Chương trình dùng Haar Cascade của OpenCV để phát hiện khuôn mặt, chuyển vùng
khuôn mặt về ảnh xám 48 × 48 và hiển thị nhãn cảm xúc cùng độ tin cậy. Nhấn
`Q` để thoát. Checkpoint lưu cấu hình CLAHE, mean và độ lệch chuẩn nên demo tự
động dùng cùng cách tiền xử lý như lúc huấn luyện.

Kết quả nhận diện chỉ là dự đoán từ biểu hiện bên ngoài của khuôn mặt, không
phải kết luận về trạng thái tâm lý của một người.

## 6. Nguyên tắc đánh giá

- Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số.
- Chọn checkpoint theo Macro-F1 trên tập validation.
- Báo cáo cả Accuracy và Macro-F1 vì FER-2013 mất cân bằng giữa các lớp.
- Dùng class weight tính từ tập train cho hàm mất mát.
- Cố định seed để tăng khả năng tái lập kết quả.
- Kết quả từ tập con phải ghi rõ là kết quả thử nghiệm sơ bộ.
- ResNet18 và demo webcam là hướng phát triển, chưa được đưa vào bảng benchmark
  hiện tại.

## 7. Cấu trúc thư mục

```text
Group30_EmotionRecognition/
|-- fetch_subset.py              # tải tập con để chạy thử
|-- prepare_data.py              # chuyển FER-2013 CSV sang NPZ
|-- prepare_image_folders.py     # chuyển bộ ảnh thư mục sang NPZ
|-- train_compare.py             # huấn luyện và so sánh ba mô hình
|-- realtime_demo.py             # nhận diện bằng webcam
|-- requirements.txt             # các thư viện cần cài đặt
|-- README.md
|-- data/                        # dữ liệu đầu vào
`-- results/                     # kết quả được tạo sau khi chạy
```

## 8. Liên kết dữ liệu

- FER-2013 gốc: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
- Bản dùng để lấy tập con: https://huggingface.co/datasets/abhilash88/fer2013-enhanced
