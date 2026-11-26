import csv
import json
import re
from collections import namedtuple
from dataclasses import asdict
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
import config as cfg

Item = namedtuple("Item", "tier path gray gt")


def roi_mask(shape, top=None, x=None):
    H, W = shape
    top = cfg.ROI_TOP if top is None else top
    x = cfg.ROI_X if x is None else x
    m = np.zeros(shape, np.uint8)
    yb = cfg.ROI_BOTTOM * H
    pts = np.array([[0, yb], [W - 1, yb], [x[1] * W, top * H], [x[0] * W, top * H]], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def line_x(rho, th, y):
    c = np.cos(th)
    return (rho - y * np.sin(th)) / c if abs(c) > 1e-6 else np.nan


def postprocess_lines(lines, shape):
    H, W = shape
    kept, kx = [], []
    for rho, th, sc in sorted(lines, key=lambda t: -t[2]):
        if abs(th - np.pi / 2) < np.deg2rad(cfg.MIN_HORIZ_DEG) or abs(np.cos(th)) < 1e-3:
            continue
        xb = line_x(rho, th, 0.9 * H)
        if any(abs(xb - x) < cfg.DEDUP * W for x in kx):
            continue
        kept.append((float(rho), float(th), float(sc)))
        kx.append(xb)
        if len(kept) == cfg.MAX_LANES:
            break
    return kept


def _images(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.with_name(p.stem + ".lines.txt").exists())


def _read(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, None, fx=cfg.SCALE, fy=cfg.SCALE, interpolation=cv2.INTER_AREA)
    gt = []
    for line in p.with_name(p.stem + ".lines.txt").read_text().splitlines():
        v = np.array(line.split(), float)
        if len(v) >= 4:
            q = v[:len(v) // 2 * 2].reshape(-1, 2) * cfg.SCALE
            q = q[q[:, 0] >= 0]
            if len(q) >= 2:
                gt.append(q)
    return g, gt


def load_items(split, max_per_tier=None, tiers=None, seed=0):
    groups = {"dev": _images(cfg.DATA / "dev")} if split == "dev" else {t: _images(cfg.DATA / "eval" / t) for t in (tiers or cfg.TIERS)}
    rng, items = np.random.default_rng(seed), []
    for tier, paths in groups.items():
        if max_per_tier and len(paths) > max_per_tier:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_per_tier, replace=False))]
        for p in paths:
            items.append(Item(tier, p, *_read(p)))
        print(f"Đã nạp {len(paths)} ảnh ({split}/{tier}).")
    if not items:
        raise SystemExit("Không tìm thấy ảnh kèm file .lines.txt trong data/. Xem README.")
    return items


def _mask(draw, shape):
    m = np.zeros(shape, np.uint8)
    draw(m)
    m[:int(cfg.ROI_TOP * shape[0])] = 0
    return m.astype(bool)


def match_lanes(lines, gt, shape):
    w = max(2, int(round(cfg.LANE_W * cfg.SCALE)))
    H = shape[0]
    y0 = int(cfg.ROI_TOP * H)
    gm = [(_mask(lambda m, q=q: cv2.polylines(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, 1, w), shape), q) for q in gt]
    gm = [(m, q) for m, q in gm if m.any()]

    def line_mask(r, t):
        x0, x1 = line_x(r, t, y0), line_x(r, t, H - 1)
        ok = np.isfinite(x0) and np.isfinite(x1) and abs(x0) + abs(x1) < 1e5
        return _mask(lambda m: cv2.line(m, (int(x0), y0), (int(x1), H - 1), 1, w) if ok else None, shape)

    pm = [line_mask(r, t) for r, t, _ in lines]
    if not gm or not pm:
        return 0, len(pm), len(gm), []
    iou = np.array([[(a & b).sum() / max((a | b).sum(), 1) for b, _ in gm] for a in pm])
    tp, lat = 0, []
    for i, j in zip(*linear_sum_assignment(-iou)):
        if iou[i, j] > cfg.IOU_THRESH:
            tp += 1
            q = gm[j][1]
            o = np.argsort(q[:, 1])
            xg = np.interp(0.9 * H, q[o, 1], q[o, 0])
            lat.append(abs(line_x(lines[i][0], lines[i][1], 0.9 * H) - xg))
    return tp, len(pm) - tp, len(gm) - tp, lat


def evaluate(fn, items):
    TP = FP = FN = 0
    lat, times, per = [], [], []
    for it in items:
        t0 = time.perf_counter()
        L = fn(it.gray)
        times.append(time.perf_counter() - t0)
        tp, fp, fn_, l = match_lanes(L, it.gt, it.gray.shape)
        TP += tp; FP += fp; FN += fn_; lat += l
        per.append(dict(item=it, lines=L, f1=2 * tp / max(2 * tp + fp + fn_, 1)))
    return dict(n=len(items), precision=TP / max(TP + FP, 1), recall=TP / max(TP + FN, 1),
                f1=2 * TP / max(2 * TP + FP + FN, 1),
                lat_px=float(np.mean(lat)) / cfg.SCALE if lat else float("nan"), ms=1000 * float(np.mean(times)), per=per)


def evaluate_by_tier(fn, items):
    return {t: evaluate(fn, [i for i in items if i.tier == t]) for t in dict.fromkeys(i.tier for i in items)}


def illum_stats(gray):
    b = gaussian_filter(gray.astype(np.float64), 8)[roi_mask(gray.shape)]
    p5, p95 = np.percentile(b, [5, 95])
    return float(b.std()), float((p95 - p5) / (p95 + p5 + 1e-9)), float(b.mean())


def write_table(name, rows):
    d = cfg.RESULTS / "tables"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Đã lưu bảng {name}.csv ({len(rows)} dòng).")


def figures_dir():
    d = cfg.RESULTS / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def overlay(item, lines, edges=None, show_gt=False, pred_thickness=3):
    g = item.gray
    im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    H = g.shape[0]
    y0 = int(cfg.ROI_TOP * H)
    if show_gt:
        for q in item.gt:
            cv2.polylines(im, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 0), 2)
    for r, t, _ in lines:
        x0, x1 = line_x(r, t, y0), line_x(r, t, H - 1)
        if np.isfinite(x0) and np.isfinite(x1) and abs(x0) + abs(x1) < 1e5:
            cv2.line(im, (int(x0), y0), (int(x1), H - 1), (0, 0, 255), pred_thickness)
    if edges is not None:
        im = np.hstack([im, cv2.cvtColor((edges * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)])
    return im


def load_params():
    f = cfg.RESULTS / "best_params.json"
    return cfg.Params(**json.loads(f.read_text())) if f.exists() else cfg.Params()


def save_params(p):
    cfg.RESULTS.mkdir(exist_ok=True)
    (cfg.RESULTS / "best_params.json").write_text(json.dumps(asdict(p), indent=2))


def video_id(path):
    p = Path(path)
    m = re.match(r"^(.*\.mp4)_\d+$", p.stem, re.I)
    return m.group(1) if m else p.parent.name
