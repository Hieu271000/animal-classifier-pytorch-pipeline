Animal Classification with Custom CNN Pipeline
Dự án này triển khai một luồng huấn luyện và suy luận hoàn chỉnh (End-to-End) cho bài toán phân loại 10 loài động vật sử dụng mạng tích chập (CNN) tùy chỉnh trên nền tảng PyTorch.

📌 Các đặc điểm nổi bật
Kiến trúc SimpleCNN tùy chỉnh: Thiết kế mạng gồm 5 khối tích chập (make_block), mỗi khối tích hợp BatchNorm và LeakyReLU để tăng tốc độ hội tụ và ổn định quá trình huấn luyện.
Luồng dữ liệu (Data Pipeline) mạnh mẽ:
  Tích hợp kỹ thuật tăng cường dữ liệu (RandomAffine, ColorJitter) để cải thiện khả năng tổng quát hóa của mô hình.
  Chuẩn hóa dữ liệu theo thông số của ImageNet để tối ưu hiệu suất.
Quản lý huấn luyện chuyên nghiệp:
  Tích hợp CLI (Command Line Interface) qua argparse cho phép cấu hình linh hoạt (epochs, batch size, learning rate...) mà không cần sửa code.
  Sử dụng TensorBoard để theo dõi trực quan hàm mất mát (Loss) và độ chính xác (Accuracy) theo thời gian thực.
  Cơ chế Checkpointing: Tự động lưu lại mô hình tốt nhất (best_cnn.pt) và mô hình cuối cùng (last_cnn.pt).
Đánh giá chi tiết: Xuất Confusion Matrix trực tiếp lên TensorBoard để phân tích sai sót giữa các lớp nhân vật.

🛠 Công nghệ sử dụng
  Framework chính: PyTorch, Torchvision.
  Xử lý hình ảnh: OpenCV, PIL (Pillow).
  Phân tích & Theo dõi: Scikit-learn, TensorBoard, Matplotlib, tqdm.
  
📂 Cấu trúc thư mục
  models.py: Chứa định nghĩa kiến trúc mạng SimpleCNN.
  dataset.py: Lớp AnimalDataset tùy chỉnh để quản lý dữ liệu từ thư mục ./data/animals/.
  train.py: Script huấn luyện hệ thống tích hợp CLI và logging.
  inference.py: Script suy luận để dự đoán hình ảnh đơn lẻ và hiển thị kết quả trực quan bằng OpenCV.
  
🚀 Hướng dẫn sử dụng
  Huấn luyện (Training)
    python train.py --root ./data --epochs 100 --batch-size 8 --image-size 224
  Suy luận (Inference)
    python inference.py --image-path path/to/your/image.jpg --checkpoint trained_models/best_cnn.pt
    
📝 Lưu ý
  Dữ liệu: Để chạy dự án, hãy đặt dữ liệu vào thư mục ./data/animals/ với cấu trúc gồm hai thư mục con train và test. Danh sách 10 lớp bao gồm: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel.
  Cấu hình: Mặc định mô hình sẽ ưu tiên sử dụng CUDA nếu máy tính của bạn có hỗ trợ GPU.
