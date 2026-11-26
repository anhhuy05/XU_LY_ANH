import cv2
import numpy as np
import config as cfg
from utils import roi_mask, postprocess_lines


def canny_fixed(gray, roi, lo=50, hi=150):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return (cv2.Canny(blur, lo, hi) > 0) & roi


def hough_standard(edges, thresh=40, d_rho=2.0, d_theta_deg=1.0, max_lines=100):
    img = edges.astype(np.uint8) * 255
    cut = np.deg2rad(cfg.MIN_HORIZ_DEG)
    out = []
    for lo, hi in [(0.0, np.pi / 2 - cut), (np.pi / 2 + cut, np.pi)]:
        r = cv2.HoughLines(img, d_rho, np.deg2rad(d_theta_deg), thresh, min_theta=lo, max_theta=hi)
        if r is not None:
            out += [(float(l[0]), float(l[1]), float(-i)) for i, l in enumerate(r[:max_lines, 0])]
    return out


def run_baseline(gray, p=None, debug=False):
    p = p or cfg.Params()
    roi = roi_mask(gray.shape)
    edges = canny_fixed(gray, roi, p.b_lo, p.b_hi)
    raw = hough_standard(edges, p.b_thresh)
    lines = postprocess_lines(raw, gray.shape)
    return (lines, dict(edges=edges)) if debug else lines
