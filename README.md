# Animal Classification with Custom CNN Pipeline

Dự án này triển khai một hệ thống phân loại động vật (10 lớp) dựa trên kiến trúc mạng tích chập (CNN) tùy chỉnh. Hệ thống được thiết kế theo dạng module chuyên nghiệp, hỗ trợ huấn luyện linh hoạt và suy luận trực quan.

## Mục lục

1. [Tổng quan Flow Code](#tổng-quan-flow-code)
2. [Yêu cầu cài đặt](#yêu-cầu-cài-đặt)
3. [Cấu hình dự án](#cấu-hình-dự-án)
4. [Cách chạy](#cách-chạy)
    - [Huấn luyện mô hình](#huấn-luyện-mô-hình)
    - [Chạy suy luận (Inference)](#chạy-suy-luận-inference)

---

## Tổng quan Flow Code

Hệ thống được chia thành nhiều module chuyên biệt, giúp quản lý luồng dữ liệu và mô hình hiệu quả:

1. **`dataset.py`**: 
    - Chịu trách nhiệm quản lý và tải dữ liệu từ thư mục `./data/animals/`.
    - Định nghĩa lớp `AnimalDataset` kế thừa từ `torch.utils.data.Dataset`.
    - Xử lý nhãn cho 10 danh mục: *butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel.

2. **`models.py`**:
    - Định nghĩa kiến trúc `SimpleCNN` với 5 khối tích chập tùy chỉnh.
    - Mỗi khối sử dụng cơ chế `BatchNorm` và `LeakyReLU` để tối ưu hóa quá trình học của mạng sâu.
    - Tích hợp lớp `Dropout` tại các tầng kết nối đầy đủ (Fully Connected) để chống quá khớp (overfitting).

3. **`train.py`**:
    - Luồng điều khiển chính cho quá trình huấn luyện.
    - Sử dụng `argparse` để nhận các tham số cấu hình từ dòng lệnh (CLI) như epochs, batch size, learning rate.
    - Tích hợp `SummaryWriter` để ghi lại nhật ký huấn luyện lên **TensorBoard**.
    - Thực hiện đánh giá mô hình qua **Confusion Matrix** sau mỗi epoch.

4. **`inference.py`**:
    - Module dự đoán kết quả cho hình ảnh đơn lẻ.
    - Tải trọng số tốt nhất (`best_cnn.pt`), xử lý ảnh đầu vào bằng OpenCV và hiển thị kết quả trực quan kèm độ tin cậy.

---

## Yêu cầu cài đặt

Để chạy dự án, bạn cần cài đặt các thư viện sau:
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib tqdm tensorboard
```

## ⚙️ Cấu hình dự án

Hệ thống hỗ trợ cấu hình linh hoạt thông qua giao diện dòng lệnh (CLI), giúp người dùng dễ dàng điều chỉnh các tham số mà không cần thay đổi mã nguồn:

*   `--root`: Đường dẫn tới thư mục gốc của bộ dữ liệu (mặc định: `./data`).
*   `--epochs`: Tổng số vòng lặp huấn luyện.
*   `--batch-size`: Số lượng mẫu dữ liệu trong một lô huấn luyện.
*   `--image-size`: Kích thước ảnh đầu vào (mặc định: 224x224).
*   `--checkpoint`: Đường dẫn tới file trọng số `.pt` đã lưu (dùng để huấn luyện tiếp hoặc chạy suy luận).

---

## 🚀 Cách chạy

### 1. Huấn luyện mô hình (Training)
Để bắt đầu quá trình huấn luyện với các tham số tùy chỉnh, bạn chạy lệnh sau:
```bash
python train.py --epochs 100 --batch-size 8 --image-size 224
```

### 2.Chạy suy luận(Inference)
```bash
python inference.py --image-path "./data/test_image.jpg" --checkpoint "trained_models/best_cnn.pt"
```
