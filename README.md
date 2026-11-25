# Group 30 - Facial Emotion Recognition on FER-2013

Du an nhap mon hoc may so sanh ba mo hinh tu don gian den phu hop voi anh:

1. `softmax`: hoi quy Softmax tren 2.304 gia tri pixel.
2. `mlp`: mang truyen thang voi mot lop an 128 nut.
3. `cnn`: CNN nho gom ba khoi tich chap.

Tat ca mo hinh dung chung cach chia du lieu, class weight va cac chi so
Accuracy, Macro-F1. Pipeline ho tro CLAHE, chuan hoa theo mean/std cua tap
train va tang cuong du lieu. Ma nguon hien tai chua dung ResNet.

## 1. Cai dat

Khuyen nghi Python 3.10-3.12:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chuan bi du lieu

### Cach A - FER-2013 CSV goc

Tai `fer2013.csv` tu trang cuoc thi FER-2013/Kaggle va dat tai:

```text
data/fer2013.csv
```

Sau do chuyen thanh tep NumPy:

```bash
python prepare_data.py --csv data/fer2013.csv --output data/fer2013.npz
```

Chuong trinh ton trong cot `Usage` cua du lieu goc:

- `Training`: huan luyen;
- `PublicTest`: xac thuc;
- `PrivateTest`: kiem thu.

### Cach B - bo anh da giai nen

Neu du lieu co cau truc `train/<class>` va `test/<class>`, chay:

```bash
python prepare_image_folders.py --root /duong-dan/archive-2 \
  --output data/fer2013_full.npz
```

`archive-2` chi la ten thu muc tai ve tren may. Script kiem tra bat buoc du
28.709 anh train, 3.589 anh validation va 3.589 anh test.

### Tap con de chay thu nhanh

Lenh sau lay mot tap con tu ban FER-2013 Enhanced cong khai tren Hugging Face.
Cach nay chi phu hop voi smoke test/bao cao tien do, khong thay the thi nghiem
tren toan bo du lieu cho bao cao cuoi ky.

```bash
python fetch_subset.py --train-size 2100 --val-size 700 --test-size 700 \
  --output data/fer2013_subset.npz
```

## 3. Chay thi nghiem

Chay ca ba mo hinh tren toan bo du lieu voi tien xu ly day du:

```bash
python train_compare.py --data data/fer2013_full.npz \
  --models softmax mlp cnn --epochs 20 --batch-size 64 \
  --clahe --standardize --augment \
  --output-dir results/full_preprocessing_run
```

Trong do:

- `--clahe`: tang tuong phan cuc bo cho anh;
- `--standardize`: tinh mean/std chi tu tap train roi chuan hoa ca ba split;
- `--augment`: chi bien doi ngau nhien anh train, gom lat ngang, xoay, zoom,
  dich chuyen va thay doi sang/toi nhe.

Ket qua cua luot chay 20 epoch tren toan bo FER-2013:

| Mo hinh | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Softmax | 24,85% | 22,26% |
| MLP | 36,53% | 30,67% |
| CNN nho | 52,08% | 46,43% |

Tien xu ly day du cai thien Accuracy cua CNN, nhung khong co loi cho Softmax.
Vi vay ket qua nay duoc luu rieng de so sanh voi benchmark khong tien xu ly.

Chay nhanh tren tap con:

```bash
python train_compare.py --data data/fer2013_subset.npz \
  --models softmax mlp cnn --epochs 5 --batch-size 64 \
  --output-dir results/progress_run
```

Ket qua gom:

- `summary.csv` va `summary.json`: Accuracy, Macro-F1, thoi gian;
- `*_confusion_matrix.png`: ma tran nham lan;
- `*_history.png`: loss va accuracy theo epoch;
- `*_best.pt`: trong so tot nhat theo validation Macro-F1;
- `run_config.json`: cau hinh de tai lap thi nghiem.

Thu muc `results/progress_run_noaug` chua ket qua 30 epoch khong augmentation
duoc trich vao bao cao tien do. Thu muc `results/progress_run` chua mot luot 15
epoch co augmentation de kiem tra nhanh anh huong cua tang cuong du lieu.

## 4. Demo webcam thoi gian thuc

Sau khi co checkpoint CNN moi, ket noi webcam va chay:

```bash
python realtime_demo.py \
  --checkpoint results/full_preprocessing_run/cnn_best.pt \
  --camera 0
```

Chuong trinh dung Haar Cascade cua OpenCV de phat hien khuon mat, dua vung mat
ve anh xam 48 x 48 va hien nhan cam xuc cung do tin cay. Nhan `Q` de thoat.
Checkpoint luu kem cau hinh CLAHE va mean/std, nen webcam se tu dong dung dung
tien xu ly nhu luc train. Nhan dien cam xuc chi la du doan tu bieu hien ben
ngoai, khong phai ket luan ve trang thai tam ly cua mot nguoi.

## 5. Nguyen tac danh gia

- Khong dung tap test de chon mo hinh.
- Chon checkpoint theo Macro-F1 tren validation.
- Bao cao ca Accuracy va Macro-F1 vi FER-2013 mat can bang lop.
- Co dinh seed de tang kha nang tai lap.
- Ket qua tap con phai ghi ro la `preliminary/subset`.

## 6. Cau truc

```text
Group30_SourceCode/
|-- fetch_subset.py
|-- prepare_data.py
|-- prepare_image_folders.py
|-- train_compare.py
|-- realtime_demo.py
|-- requirements.txt
|-- README.md
|-- data/                  # du lieu khong dua vao ZIP
`-- results/               # ket qua duoc tao sau khi chay
```

## 7. Lien ket du lieu

- FER-2013 goc: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
- Ban dung de lay tap con: https://huggingface.co/datasets/abhilash88/fer2013-enhanced
