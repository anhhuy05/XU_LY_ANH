# Lane_detection: phát hiện vạch kẻ đường trong điều kiện ánh sáng không đều

Môn Xử lý ảnh và Thị giác máy tính (121036), bài tập lớn. Thách thức C: ánh sáng không đều (bóng đổ, vùng sáng gắt).

- **Baseline**: Gauss, Canny ngưỡng cố định, Hough chuẩn tắc (OpenCV), file `baseline.py`.
- **Phương pháp đề xuất P**: chuẩn hóa chiếu sáng I/m (ảnh tích phân), Canny tự viết với ngưỡng tương đối, Hough bỏ phiếu theo hướng gradient, file `proposed.py`.

## Cấu trúc
```
Lane_detection/
├── baseline.py            canny_fixed, hough_standard, run_baseline
├── proposed.py            illumination_norm, canny_adaptive, hough_gradient, run_proposed
├── utils.py               ROI, nạp dữ liệu, F1 kiểu CULane, vẽ ảnh
├── config.py              hằng số, Params (cờ use_norm / use_vote)
├── run_all.py             chạy toàn bộ: phân tầng, E2, E1, E3, E4
├── experiments/
│   ├── e1_challenge.py          E1 chứng cứ thách thức (baseline theo tầng, ảnh hỏng)
│   ├── e2_params.py             E2 chọn tham số theo F1 ưu tiên shadow trên dev
│   ├── e3_ablation.py           E3 loại bỏ thành phần
│   ├── e4_breaking_point.py     E4 điểm gãy khi tăng độ sâu bóng
│   ├── compare_images.py        ảnh so sánh baseline và P cho báo cáo
│   └── export_all.py            xuất ảnh đã xử lý của mọi ảnh
├── tools/
│   ├── make_stratification.py   ghi data/stratification.csv (tầng, ánh sáng, video, kiểm tra rò rỉ)
│   └── resplit_data.py          chia lại dev/eval theo video, chia tầng theo độ không đều ánh sáng
├── data/                  ảnh và nhãn CULane (xem data/README.md)
└── results/               bảng, hình (tự tạo khi chạy)
```

## Cài đặt
```
pip install -r requirements.txt
```
Yêu cầu Python 3.9 trở lên.

## Dữ liệu và cách chạy
Bộ dữ liệu sử dụng là [CULane](https://xingangpan.github.io/projects/CULane.html). Tải dữ liệu từ liên kết Google Drive hoặc Baidu Cloud trên trang chính thức, giải nén vào `data/raw/culane/`, rồi chia tập phát triển và đánh giá theo video:
```
python -m tools.resplit_data --src data/raw/culane --out data/prepared
```
Để chạy trên tập đã chia, đặt `$env:LANE_DATA="data/prepared"` trong PowerShell hoặc `LANE_DATA=data/prepared` trên Linux/macOS. Mọi đường dẫn dữ liệu phải nằm trong `data/`.

CULane được cung cấp cho nghiên cứu/giảng dạy phi thương mại. Khi sử dụng, trích dẫn công trình CULane và không phân phối lại bộ dữ liệu; xem điều khoản trên trang chính thức. Nếu đã có sẵn ảnh/nhãn, đặt chúng theo cấu trúc trong [data/README.md](data/README.md).

Chạy toàn bộ thí nghiệm (mặc định duyệt mọi ảnh):
```
python run_all.py
```
Hoặc chạy từng bước; E2 cần chạy trước vì ghi `results/best_params.json` cho các bước sau:
```
python -m tools.make_stratification
python -m experiments.e2_params
python -m experiments.e1_challenge
python -m experiments.e3_ablation
python -m experiments.e4_breaking_point
python -m experiments.compare_images --k 5 --all
python -m experiments.export_all
```
Thêm `--max N` để chạy N ảnh mỗi tầng. Các bảng, hình và ảnh kết quả được ghi vào `results/`.

Tạo video so sánh baseline và P từ `data/demo/road.mp4`:
```
python -m experiments.demo_video
```
Video gọn (đường mạnh nhất mỗi bên) lưu tại `results/demo/road_comparison.mp4`;
bản giữ tất cả dự đoán lưu tại `results/demo/road_comparison_all_lines.mp4`.

## Kết quả tạo ra
| File | Nội dung |
|---|---|
| `data/stratification.csv` | từng ảnh: tầng, video, `illum_std`, `illum_ratio`, số vạch nhãn |
| `results/tables/e1_baseline.csv` | E1: chỉ số baseline theo tầng |
| `results/failure_cases/` | E1: ảnh baseline hỏng điển hình |
| `results/tables/e2_sweep.csv`, `figures/e2_curves.png`, `e2_intermediate.png` | E2: khảo sát tham số, ảnh trung gian |
| `results/tables/e3_ablation.csv`, `figures/e3_ablation.png` | E3: loại bỏ thành phần |
| `results/tables/e4_break.csv`, `figures/e4_break.png` | E4: điểm gãy |
| `results/comparison/`, `results/all_images/`, `tables/per_image.csv` | ảnh so sánh và ảnh đã xử lý |

E2 chỉ chọn tham số trên dev theo điểm shadow-robust: `0.5 × F1_shadow + 0.25 × F1 macro theo video trên nhóm 25% ảnh có illum_ratio cao nhất + 0.25 × F1 trung bình các tầng khác`. Việc chọn cấu hình P bị ràng buộc thêm: F1 shadow phải ít nhất bằng baseline trên dev. Tìm kiếm bắt đầu từ cấu hình P tốt trước đó để tránh coordinate sweep bỏ lỡ điểm khởi đầu tốt; bảng E2 ghi cột `eligible` cho biết cấu hình P có đạt ràng buộc hay không. Cách tính theo video giảm ảnh hưởng của nhiều frame gần nhau trong cùng clip; bảng E2 vẫn ghi F1 theo tầng để thấy rõ đánh đổi.
P có ROI phát hiện riêng để giới hạn cạnh ngoài vùng làn; ROI dùng để chấm điểm vẫn cố định, không thay đổi theo tham số của P.

## Dữ liệu
- Nguồn: CULane, X. Pan, J. Shi, P. Luo, X. Wang và X. Tang, "Spatial as deep: Spatial CNN for traffic scene understanding," in Proc. AAAI Conf. Artif. Intell., 2018. Xem liên kết tải và giấy phép trên [trang chính thức CULane](https://xingangpan.github.io/projects/CULane.html).
- Nhãn: file `.lines.txt` của CULane, mỗi dòng là một vạch kẻ do người gán.
- Cường độ của C đo bằng `illum_ratio` = (p95 - p5)/(p95 + p5) của ảnh làm mờ mạnh trong ROI, và `illum_std`. Tầng nhẹ, vừa, nặng chia theo ba phân vị của `illum_ratio` (khi dùng `resplit_data`) hoặc theo thư mục CULane `normal`, `hlight`, `shadow`.
- Tập dev (chọn tham số) và eval (đánh giá) tách theo video: `make_stratification` in số video xuất hiện ở cả hai tập, phải bằng 0.

## Thước đo
F1 theo kiểu CULane: mỗi vạch được vẽ thành dải rộng 30 px (ở ảnh gốc), IoU lớn hơn 0.5 là khớp, ghép bằng thuật toán Hungarian. Kèm precision, recall, sai số ngang tại hàng y = 0.9H và thời gian xử lý mỗi ảnh. Ảnh được thu nhỏ còn 0.5 (`config.SCALE`).

## Mã nguồn tự viết và hàm dựng sẵn
- Tự viết: ảnh tích phân, chuẩn hóa chiếu sáng, Sobel, triệt phi cực đại, ngưỡng kép, bỏ phiếu Hough theo gradient, chọn đỉnh trên mảng tích lũy, thước đo F1.
- Dùng thư viện: OpenCV chỉ dùng cho baseline (`cv2.Canny`, `cv2.HoughLines`) và các tiện ích (đọc ảnh, vẽ); SciPy dùng cho làm mờ Gauss, lọc cực đại và ghép Hungarian.
