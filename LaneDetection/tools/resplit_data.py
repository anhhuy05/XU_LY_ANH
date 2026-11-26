import sys
import csv
import shutil
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import numpy as np
from utils import _images, _read, illum_stats, video_id

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="thư mục ảnh nguồn bên trong data/")
ap.add_argument("--out", default="data/resplit", help="thư mục đích bên trong data/")
ap.add_argument("--dev-frac", type=float, default=0.3)
ap.add_argument("--per-tier", type=int, default=100)
ap.add_argument("--dev-size", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

try:
    src, out = cfg.resolve_data_path(args.src), cfg.resolve_data_path(args.out)
except ValueError as exc:
    ap.error(str(exc))
if src == out or src in out.parents or out in src.parents:
    ap.error("--src và --out phải là các thư mục không lồng nhau")
paths = _images(src)
if not paths:
    raise SystemExit("Không thấy ảnh kèm .lines.txt trong --src")
rng = np.random.default_rng(args.seed)
vids = [video_id(p) for p in paths]
group = list(vids)
if len(set(vids)) < 4:
    counter = {}
    for i, (p, v) in enumerate(zip(paths, vids)):
        counter[v] = counter.get(v, -1) + 1
        group[i] = f"{v}#{counter[v] // 20}"
ug = sorted(set(group))
rng.shuffle(ug)
n_dev = max(1, int(round(args.dev_frac * len(ug))))
dev_groups = set(ug[:n_dev])
print(f"Dữ liệu: {len(paths)} ảnh, {len(set(vids))} video, {len(ug)} nhóm "
      f"(dev={n_dev}, eval={len(ug) - n_dev}).")

stats = {}
for i, p in enumerate(paths):
    g, _ = _read(p)
    stats[p] = illum_stats(g)
    if (i + 1) % 200 == 0:
        print(f"Đang đo sáng: {i + 1}/{len(paths)} ảnh.", flush=True)

dev = [p for p, g in zip(paths, group) if g in dev_groups]
ev = [p for p, g in zip(paths, group) if g not in dev_groups]
if len(ev) < 3:
    raise SystemExit("Quá ít ảnh eval")
r = np.array([stats[p][1] for p in ev])
q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
tier_of = {p: ("light" if stats[p][1] <= q1 else "medium" if stats[p][1] <= q2 else "heavy") for p in ev}


def pick(lst, n):
    if len(lst) <= n:
        return list(lst)
    return [lst[i] for i in sorted(rng.choice(len(lst), n, replace=False))]


chosen = [("dev", "dev", p) for p in pick(dev, args.dev_size)]
for t in ["light", "medium", "heavy"]:
    chosen += [("eval", t, p) for p in pick([p for p in ev if tier_of[p] == t], args.per_tier)]


def new_stem(p):
    v = video_id(p)
    return p.stem if p.stem.lower().startswith(v.lower()) else f"{p.parent.name}_{p.stem}"


rows = []
for split, tier, p in chosen:
    d = out / "dev" if split == "dev" else out / "eval" / tier
    d.mkdir(parents=True, exist_ok=True)
    ns = new_stem(p)
    shutil.copy(p, d / (ns + p.suffix))
    shutil.copy(p.with_name(p.stem + ".lines.txt"), d / (ns + ".lines.txt"))
    s = stats[p]
    rows.append(dict(image=str((d / (ns + p.suffix)).relative_to(out)), split=split, tier=tier, video=video_id(p),
                     illum_std=round(s[0], 2), illum_ratio=round(s[1], 3), mean_lum=round(s[2], 1)))
with open(out / "stratification.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for split, tier in [("dev", "dev"), ("eval", "light"), ("eval", "medium"), ("eval", "heavy")]:
    x = [a for a in rows if a["split"] == split and a["tier"] == tier]
    if x:
        print(f"{split}/{tier}: {len(x)} ảnh | illum_ratio={np.mean([a['illum_ratio'] for a in x]):.3f}.")
print(f"Ngưỡng illum_ratio: nhẹ≤{q1:.3f}, vừa≤{q2:.3f}, còn lại=nặng.")
print(f"Đã lưu dữ liệu tại {out}; chọn bằng LANE_DATA.")
