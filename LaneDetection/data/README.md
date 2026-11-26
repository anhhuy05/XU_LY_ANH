# Dữ liệu

Đặt ảnh và nhãn CULane cùng thư mục, theo cấu trúc:

```text
data/
├── raw/                    # nguồn CULane bổ sung, không dùng trực tiếp khi đánh giá
├── danh_sach_tang.csv      # nhật ký ảnh được gán vào dev/eval và tầng
├── stratification.csv      # tầng, video và chỉ số ánh sáng của từng ảnh
├── dev/
│   └── [<tầng>/]
│       ├── <ảnh>.jpg
│       └── <ảnh>.lines.txt
└── eval/
    └── <tầng>/
        ├── <ảnh>.jpg
        └── <ảnh>.lines.txt
```

Ảnh hỗ trợ định dạng JPG, JPEG và PNG. Tên nhãn phải khớp tên ảnh với đuôi
`.lines.txt`. Mọi bộ dữ liệu nguồn và dữ liệu đã chia lại phải nằm dưới thư mục
này. Đường dẫn `LANE_DATA`, nếu đặt, cũng phải trỏ vào đây.

Bộ dữ liệu dùng trong bài là CULane. Xem trang chính thức để tải dữ liệu và đọc
điều khoản sử dụng: https://xingangpan.github.io/projects/CULane.html

Kết quả chạy thí nghiệm được lưu trong `../results/`.
