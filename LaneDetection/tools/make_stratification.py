import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config as cfg
from utils import _images, _read, illum_stats, video_id

groups = [("dev", "dev", _images(cfg.DATA / "dev"))] + [("eval", t, _images(cfg.DATA / "eval" / t)) for t in cfg.TIERS]
rows = []
for split, tier, paths in groups:
    for p in paths:
        g, gt = _read(p)
        s = illum_stats(g)
        rows.append(dict(image=str(p.relative_to(cfg.DATA)), split=split, tier=tier, video=video_id(p),
                         illum_std=round(s[0], 2), illum_ratio=round(s[1], 3), mean_lum=round(s[2], 1), n_gt_lanes=len(gt)))
if not rows:
    raise SystemExit("Không có ảnh kèm .lines.txt trong data/")
out = cfg.DATA / "stratification.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"Đã lưu bảng phân tầng: {out}")
for split, tier, _ in groups:
    r = [x for x in rows if x["split"] == split and x["tier"] == tier]
    if r:
        print(f"{split}/{tier}: {len(r)} ảnh, {len({x['video'] for x in r})} video | "
              f"illum_ratio={np.mean([x['illum_ratio'] for x in r]):.3f}, "
              f"illum_std={np.mean([x['illum_std'] for x in r]):.1f}.")
