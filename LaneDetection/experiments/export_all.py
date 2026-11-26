import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import config as cfg
from utils import load_items, overlay, load_params, match_lanes, write_table
from baseline import run_baseline
from proposed import run_proposed

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None)
ap.add_argument("--dev", action="store_true")
args = ap.parse_args()
P = load_params()
out = cfg.RESULTS / "all_images"


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def f1_of(lines, it):
    tp, fp, fn, _ = match_lanes(lines, it.gt, it.gray.shape)
    return 2 * tp / max(2 * tp + fp + fn, 1), (tp, fp, fn)


def edge_img(e):
    return cv2.cvtColor((e * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def process(it):
    lb, db = run_baseline(it.gray, P, True)
    lp, dp = run_proposed(it.gray, P, True)
    fb, cb = f1_of(lb, it)
    fp_, cp = f1_of(lp, it)
    n = dp["norm"]
    lo, hi = np.percentile(n, [1, 99])
    norm = cv2.cvtColor(np.clip((n - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    top = [label(cv2.cvtColor(it.gray, cv2.COLOR_GRAY2BGR), f"Original ({it.tier})"),
           label(overlay(it, lb), f"Baseline F1={fb:.2f} TP/FP/FN={cb[0]}/{cb[1]}/{cb[2]}"),
           label(overlay(it, lp), f"Proposed F1={fp_:.2f} TP/FP/FN={cp[0]}/{cp[1]}/{cp[2]}")]
    bot = [label(norm, "After illumination normalization"), label(edge_img(db["edges"]), "Baseline edges"),
           label(edge_img(dp["edges"]), "Proposed edges")]
    return np.vstack([np.hstack(top), np.hstack(bot)]), fb, fp_


items = load_items("eval", args.max)
if args.dev:
    items += load_items("dev", args.max)
rows = []
for k, it in enumerate(items):
    folder = out / it.tier
    folder.mkdir(parents=True, exist_ok=True)
    img, fb, fp_ = process(it)
    cv2.imwrite(str(folder / f"{it.path.stem}.png"), img)
    rows.append(dict(tier=it.tier, image=it.path.name, baseline_f1=round(fb, 3), proposed_f1=round(fp_, 3)))
    if (k + 1) % 20 == 0 or k + 1 == len(items):
        print(f"Tiến độ: {k + 1}/{len(items)} ảnh.", flush=True)
write_table("per_image", rows)
print(f"Đã lưu ảnh xử lý: {out}")
