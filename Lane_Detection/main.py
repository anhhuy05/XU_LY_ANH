import cv2
import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "road.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "output.mp4")

SHOW_WINDOWS = True         # Đặt False nếu chạy trên server/Colab không có giao diện
USE_DEFAULT_LANES = False   # True: khi mất vạch thì vẽ làn mặc định (màu đỏ để phân biệt)

ROI_TOP_RATIO = 0.50        # Dùng chung cho ROI và tracker
GRAY_THRESHOLD = 180        # Ngưỡng vùng sáng trên ảnh đã qua CLAHE

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# ==========================================
# 2. CẢI THIỆN ÁNH SÁNG BẰNG CLAHE
# ==========================================

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(gray):
    return _clahe.apply(gray)


# ==========================================
# 3. KHU VỰC MẶT ĐƯỜNG (ROI)
# ==========================================

def region_of_interest(image):

    height, width = image.shape

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.08), height),
        (int(width * 0.43), int(height * ROI_TOP_RATIO)),
        (int(width * 0.57), int(height * ROI_TOP_RATIO)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


# ==========================================
# 4. TẠO MASK VẠCH KẺ ĐƯỜNG
# ==========================================

def create_lane_mask(frame, enhanced):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Vạch trắng: độ bão hòa thấp, độ sáng cao
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 150], dtype=np.uint8),
        np.array([179, 75, 255], dtype=np.uint8)
    )

    # Vạch vàng
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 55, 80], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8)
    )

    # Vùng sáng trên ảnh đã cân bằng bằng CLAHE
    bright_mask = cv2.inRange(enhanced, GRAY_THRESHOLD, 255)

    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    lane_mask = cv2.bitwise_or(lane_mask, bright_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)   # bỏ chấm nhiễu nhỏ
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)  # nối các đoạn đứt nhỏ

    return lane_mask


# ==========================================
# 5. THEO DÕI LÀN GIỮA CÁC KHUNG HÌNH
# ==========================================

class LaneTracker:
    """
    Mỗi biên làn là đường thẳng: x = a * (y - y_start) + b
    coeff = [a, b]
    """

    BAND_COUNT = 14
    MIN_POINTS = 5
    MIN_FILL_RATIO = 0.15        # tỉ lệ pixel vạch tối thiểu trong 1 cột của dải
    FIRST_WINDOW_RATIO = 0.16    # cửa sổ tìm ở dải đầu tiên (sát đáy)
    NEXT_WINDOW_RATIO = 0.07     # cửa sổ tìm ở các dải sau (vạch dịch ít giữa 2 dải)
    INLIER_TOL_RATIO = 0.02      # sai số tối đa của điểm so với đường fit
    MAX_JUMP_RATIO = 0.08        # lệch tối đa ở đáy so với khung trước
    MIN_LANE_WIDTH_RATIO = 0.25  # độ rộng làn hợp lệ ở đáy khung hình
    MAX_LANE_WIDTH_RATIO = 0.85
    MAX_MISSING = 10             # số khung tối đa giữ vạch cũ khi mất vạch
    ALPHA = 0.35                 # hệ số làm mượt

    def __init__(self):
        self.coeffs = {"left": None, "right": None}
        self.missing = {"left": 0, "right": 0}

    def reset(self):
        self.coeffs = {"left": None, "right": None}
        self.missing = {"left": 0, "right": 0}

    # ---------- tìm điểm vạch theo từng dải ----------

    def _detect(self, mask, side):
        """Trả về hệ số đường thẳng mới (chưa làm mượt) hoặc None."""

        height, width = mask.shape
        y_start = int(height * ROI_TOP_RATIO)
        y_end = height - 8
        band_height = max(8, (y_end - y_start) // self.BAND_COUNT)

        previous = self.coeffs[side]

        if previous is not None:
            expected_x = int(np.polyval(previous, y_end - y_start))
        else:
            expected_x = int(width * (0.28 if side == "left" else 0.72))

        # Mỗi bên chỉ tìm trong nửa khung hình của nó
        if side == "left":
            side_min, side_max = 0, int(width * 0.48)
        else:
            side_min, side_max = int(width * 0.52), width

        points = []

        for band_index in range(self.BAND_COUNT):

            y2 = y_end - band_index * band_height
            y1 = max(y_start, y2 - band_height)

            if y2 <= y1:
                continue

            window = int(width * (self.FIRST_WINDOW_RATIO if not points
                                  else self.NEXT_WINDOW_RATIO))

            search_start = max(side_min, expected_x - window)
            search_end = min(side_max, expected_x + window)

            if search_end <= search_start:
                continue

            # Đếm số pixel vạch theo từng cột (mask 0/255 -> 0/1)
            histogram = (mask[y1:y2, :] > 0).sum(axis=0).astype(np.float32)

            segment = histogram[search_start:search_end]
            peak_value = float(np.max(segment))

            if peak_value < (y2 - y1) * self.MIN_FILL_RATIO:
                continue

            peak_x = search_start + int(np.argmax(segment))

            # Lấy trọng tâm quanh đỉnh để chính xác hơn
            nx1 = max(search_start, peak_x - 18)
            nx2 = min(search_end, peak_x + 19)
            xs_local = np.arange(nx1, nx2)
            weights = histogram[nx1:nx2]

            if weights.sum() > 0:
                detected_x = int((xs_local * weights).sum() / weights.sum())
            else:
                detected_x = peak_x

            points.append(((y1 + y2) // 2, detected_x))
            expected_x = detected_x

        if len(points) < self.MIN_POINTS:
            return None

        ys = np.array([p[0] - y_start for p in points], dtype=np.float32)
        xs = np.array([p[1] for p in points], dtype=np.float32)

        # Fit đường thẳng, loại điểm ngoại lai rồi fit lại
        try:
            coeff = np.polyfit(ys, xs, 1)

            residual = np.abs(np.polyval(coeff, ys) - xs)
            inliers = residual <= width * self.INLIER_TOL_RATIO

            if inliers.sum() < self.MIN_POINTS:
                return None

            if inliers.sum() < len(ys):
                coeff = np.polyfit(ys[inliers], xs[inliers], 1)

        except (np.linalg.LinAlgError, ValueError):
            return None

        # So với khung trước: đáy khung hình không được nhảy quá xa
        if previous is not None:
            bottom_new = np.polyval(coeff, y_end - y_start)
            bottom_old = np.polyval(previous, y_end - y_start)

            if abs(bottom_new - bottom_old) > width * self.MAX_JUMP_RATIO:
                return None

        return coeff

    # ---------- cập nhật trạng thái ----------

    def update(self, mask, height, width):
        """
        Trả về dict: side -> (points, is_default)
        """

        y_start = int(height * ROI_TOP_RATIO)
        y_end = height - 8
        span = y_end - y_start

        candidates = {
            "left": self._detect(mask, "left"),
            "right": self._detect(mask, "right")
        }

        # Kiểm tra độ rộng làn ở đáy khung hình
        if candidates["left"] is not None and candidates["right"] is not None:
            lane_width = (np.polyval(candidates["right"], span)
                          - np.polyval(candidates["left"], span))

            if not (width * self.MIN_LANE_WIDTH_RATIO
                    <= lane_width
                    <= width * self.MAX_LANE_WIDTH_RATIO):
                candidates = {"left": None, "right": None}

        for side in ("left", "right"):

            candidate = candidates[side]

            if candidate is None:
                self.missing[side] += 1

                if self.missing[side] > self.MAX_MISSING:
                    self.coeffs[side] = None

            else:
                previous = self.coeffs[side]

                if previous is not None:
                    candidate = previous * (1 - self.ALPHA) + candidate * self.ALPHA

                self.coeffs[side] = candidate
                self.missing[side] = 0

        # Tạo điểm để vẽ
        y_values = np.arange(y_start, y_end + 1)
        y_int = y_values.astype(np.int32)

        defaults = {
            "left": np.array([(width * 0.22 - width * 0.48) / span, width * 0.48]),
            "right": np.array([(width * 0.78 - width * 0.52) / span, width * 0.52])
        }

        result = {}

        for side in ("left", "right"):

            coeff = self.coeffs[side]
            is_default = False

            if coeff is None:
                if not USE_DEFAULT_LANES:
                    continue
                coeff = defaults[side]
                is_default = True

            x_values = np.polyval(coeff, y_values - y_start)
            x_values = np.clip(x_values, 0, width - 1).astype(np.int32)

            result[side] = (np.column_stack((x_values, y_int)), is_default)

        return result


lane_tracker = LaneTracker()


# ==========================================
# 6. VẼ KẾT QUẢ
# ==========================================

def draw_result(frame, lanes):

    result = frame.copy()

    left = lanes.get("left")
    right = lanes.get("right")

    # Tô vùng làn đường khi cả hai biên đều là vạch thật
    if left is not None and right is not None and not left[1] and not right[1]:

        overlay = result.copy()
        polygon = np.vstack((left[0], right[0][::-1])).astype(np.int32)

        cv2.fillPoly(overlay, [polygon], (0, 180, 0))
        result = cv2.addWeighted(overlay, 0.25, result, 0.75, 0)

    for points, is_default in lanes.values():

        color = (0, 0, 255) if is_default else (0, 220, 255)

        cv2.polylines(
            result,
            [points],
            isClosed=False,
            color=color,
            thickness=4,
            lineType=cv2.LINE_AA
        )

    status = "LANE: OK" if len(lanes) == 2 else "LANE: LOST"
    cv2.putText(result, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return result


# ==========================================
# 7. PHÁT HIỆN VẠCH KẺ ĐƯỜNG
# ==========================================

def detect_lanes(frame):

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = apply_clahe(gray)

    lane_mask = create_lane_mask(frame, enhanced)
    lane_mask_roi = region_of_interest(lane_mask)

    lanes = lane_tracker.update(lane_mask_roi, height, width)

    result = draw_result(frame, lanes)

    return enhanced, lane_mask_roi, result


# ==========================================
# 8. HIỂN THỊ CÁC CỬA SỔ
# ==========================================

def show_image(window_name, image):

    max_width = 640
    max_height = 360

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)

    resized = cv2.resize(image, (int(width * scale), int(height * scale)))

    cv2.imshow(window_name, resized)


# ==========================================
# 9. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Cannot open video!")
        print("Check video path:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    writer = None
    lane_tracker.reset()

    print("Running...")
    if SHOW_WINDOWS:
        print("Press Q or ESC to exit.")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            enhanced, binary, result = detect_lanes(frame)

            if writer is None:

                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

                if not writer.isOpened():
                    print("Cannot create output video!")
                    writer = None
                    break

            writer.write(result)

            if SHOW_WINDOWS:
                show_image("1. VIDEO GOC", frame)
                show_image("2. VIDEO SAU CLAHE", enhanced)
                show_image("3. MASK VACH (SAU ROI)", binary)
                show_image("4. VIDEO NHAN DIEN VACH KE", result)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if SHOW_WINDOWS:
            cv2.destroyAllWindows()

    print("Completed!")
    print("Output video:", ascii(OUTPUT_PATH))


if __name__ == "__main__":
    main()