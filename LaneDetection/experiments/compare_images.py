import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import config as cfg
from utils import load_items, evaluate, overlay, load_params, _read, Item
from baseline import run_baseline
from proposed import run_proposed
from utils import match_lanes

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--image", default=None)
ap.add_argument("--all", action="store_true", help="vẫn xuất k ảnh mỗi tầng (ưu tiên ảnh baseline hỏng nhất) kể cả khi P không hơn baseline")
args = ap.parse_args()
P = load_params()
out = cfg.RESULTS / "comparison"; out.mkdir(parents=True, exist_ok=True)


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def f1_of(lines, it):
    tp, fp, fn, _ = match_lanes(lines, it.gt, it.gray.shape)
    return 2 * tp / max(2 * tp + fp + fn, 1), (tp, fp, fn)


def make(it, name):
    lb, db = run_baseline(it.gray, P, True)
    lp, dp = run_proposed(it.gray, P, True)
    fb, cb = f1_of(lb, it); fp_, cp = f1_of(lp, it)
    n = dp["norm"]; lo, hi = np.percentile(n, [1, 99])
    norm = cv2.cvtColor(np.clip((n - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    edge = lambda e: cv2.cvtColor((e * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    top = [label(cv2.cvtColor(it.gray, cv2.COLOR_GRAY2BGR), f"Original ({it.tier})"),
           label(overlay(it, lb), f"Baseline F1={fb:.2f} TP/FP/FN={cb[0]}/{cb[1]}/{cb[2]}"),
           label(overlay(it, lp), f"Proposed F1={fp_:.2f} TP/FP/FN={cp[0]}/{cp[1]}/{cp[2]}")]
    bot = [label(norm, "After illumination normalization"), label(edge(db["edges"]), "Baseline edges"), label(edge(dp["edges"]), "Proposed edges")]
    cv2.imwrite(str(out / name), np.vstack([np.hstack(top), np.hstack(bot)]))
    print(f"Đã lưu {name} | F1 baseline={fb:.2f}, P={fp_:.2f}.")


if args.image:
    try:
        p = cfg.resolve_data_path(args.image)
    except ValueError as exc:
        ap.error(str(exc))
    if not p.is_file():
        ap.error(f"Không tìm thấy ảnh trong data/: {p}")
    g, gt = _read(p)
    make(Item("custom", p, g, gt), f"compare_{p.stem}.png")
else:
    items = load_items("eval", args.max)
    pb = evaluate(lambda g: run_baseline(g, P), items)["per"]
    pp = evaluate(lambda g: run_proposed(g, P), items)["per"]
    for tier in dict.fromkeys(i.tier for i in items):
        idx = [i for i, it in enumerate(items) if it.tier == tier]
        idx.sort(key=lambda i: pp[i]["f1"] - pb[i]["f1"], reverse=True)
        good = [i for i in idx if pp[i]["f1"] > pb[i]["f1"]][:args.k]
        if not good and args.all:
            good = sorted(idx, key=lambda i: (pb[i]["f1"], -pp[i]["f1"]))[:args.k]
            print(f"{tier}: P chưa hơn baseline; đã lưu {len(good)} ảnh baseline kém nhất.")
        elif not good:
            print(f"{tier}: P chưa hơn baseline; không có ảnh được lưu. Dùng --all hoặc --image.")
        for rank, i in enumerate(good):
            make(items[i], f"compare_{tier}_{rank}.png")
