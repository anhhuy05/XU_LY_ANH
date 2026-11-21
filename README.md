# XU_LY_ANH 

# 🚦 ĐẾM XE MÁY ĐI QUA CAMERA GIAO THÔNG CỐ ĐỊNH TRONG ĐIỀU KIỆN BỊ CHE KHUẤT MỘT PHẦN

## 1. Giới thiệu đề tài

Đề tài xây dựng một hệ thống xử lý ảnh và thị giác máy tính nhằm phát hiện, theo dõi và đếm số lượng xe máy di chuyển qua camera giao thông cố định.

Trong thực tế, tại các tuyến đường đông đúc, xe máy thường xuyên bị che khuất một phần bởi các phương tiện khác, gây khó khăn cho việc phát hiện và đếm chính xác.

Để giải quyết vấn đề này, đề tài ứng dụng mô hình YOLO kết hợp với kỹ thuật theo dõi đối tượng và xử lý video bằng OpenCV nhằm nâng cao khả năng phát hiện và đếm xe máy trong điều kiện giao thông phức tạp.

## 2. Mục tiêu đề tài

- Phát hiện xe máy xuất hiện trong video giao thông.
- Theo dõi chuyển động của từng xe máy qua các khung hình.
- Đếm số lượng xe máy đi qua một đường đếm xác định.
- Hạn chế tình trạng đếm trùng một phương tiện.
- Khảo sát khả năng phát hiện xe máy khi bị che khuất một phần.
- Đánh giá hiệu quả của hệ thống thông qua kết quả thực nghiệm.

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| YOLO | Phát hiện phương tiện giao thông |
| OpenCV | Xử lý hình ảnh và video |
| NumPy | Xử lý dữ liệu và ma trận |
| Object Tracking | Theo dõi phương tiện qua các khung hình |
| Matplotlib | Trực quan hóa kết quả |

## 4. Nguyên lý hoạt động

Hệ thống hoạt động theo quy trình:

1. **Đầu vào:** Tiếp nhận video từ camera giao thông cố định.
2. **Tiền xử lý:** Đọc và xử lý từng khung hình bằng OpenCV.
3. **Phát hiện đối tượng:** Sử dụng YOLO để nhận diện xe máy.
4. **Theo dõi đối tượng:** Gán ID cho từng phương tiện nhằm duy trì định danh qua các khung hình.
5. **Xác định đường đếm:** Thiết lập một đường ảo trên video.
6. **Đếm phương tiện:** Khi xe máy đi qua đường ảo, hệ thống ghi nhận số lượng xe.
7. **Hiển thị kết quả:** Xuất video có khung nhận diện, ID phương tiện và tổng số xe máy đã đếm.

