import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import config as cfg
from utils import load_items, evaluate_by_tier, illum_stats, write_table, overlay, load_params
from baseline import run_baseline

ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None)
args = ap.parse_args()
P = load_params()
items = load_items("eval", args.max)
res = evaluate_by_tier(lambda g: run_baseline(g, P), items)

rows = []
for t, r in res.items():
    st = np.array([illum_stats(i.gray) for i in items if i.tier == t])
    rows.append(dict(tier=t, n=r["n"], precision=round(r["precision"], 3), recall=round(r["recall"], 3), f1=round(r["f1"], 3),
                     lat_px=round(r["lat_px"], 1), illum_std=round(float(st[:, 0].mean()), 2), illum_ratio=round(float(st[:, 1].mean()), 3),
                     mean_lum=round(float(st[:, 2].mean()), 1), mean_gt_lanes=round(float(np.mean([len(i.gt) for i in items if i.tier == t])), 2)))
write_table("e1_baseline", rows)

out = cfg.RESULTS / "failure_cases"; out.mkdir(parents=True, exist_ok=True)
for t, r in res.items():
    for k, d in enumerate(sorted(r["per"], key=lambda z: z["f1"])[:3]):
        _, dbg = run_baseline(d["item"].gray, P, debug=True)
        cv2.imwrite(str(out / f"e1_{t}_{k}_f1_{d['f1']:.2f}.png"), overlay(d["item"], d["lines"], dbg["edges"]))
print(f"Đã lưu ảnh lỗi baseline: {out}")
