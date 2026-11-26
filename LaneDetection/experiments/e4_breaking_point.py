import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_items, evaluate, write_table, figures_dir, load_params, Item, _mask
from baseline import run_baseline
from proposed import run_proposed

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None)
ap.add_argument("--noise", type=float, default=2.0, help="sigma nhiễu cảm biến")
ap.add_argument("--k", type=float, default=3.0, help="ngưỡng CNR dùng trong dự đoán")
args = ap.parse_args()
if args.max is not None and args.max < 1:
    ap.error("--max phải lớn hơn 0")
if not np.isfinite(args.noise) or args.noise < 0:
    ap.error("--noise phải là số hữu hạn không âm")
if not np.isfinite(args.k) or args.k <= 0:
    ap.error("--k phải là số hữu hạn lớn hơn 0")

P = load_params()
items = load_items("eval", args.max, tiers=["normal"])
DEPTHS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
EDGES = {"cứng": 1.0, "mềm": 25.0}


def make_shadow(shape, seed, edge):
    r = np.random.default_rng(seed); H, W = shape
    a = r.uniform(0, 2 * np.pi); cx, cy = r.uniform(.3, .7) * W, r.uniform(.6, .9) * H
    yy, xx = np.mgrid[:H, :W]
    s = np.cos(a) * (xx - cx) + np.sin(a) * (yy - cy)
    return 1 / (1 + np.exp(-np.clip(s / max(edge, .3), -50, 50)))


def apply_shadow(g, d, m, seed):
    out = g.astype(float) * (1 - d * m) + np.random.default_rng(seed + 999).normal(0, args.noise, g.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def lane_contrast(it):
    m = _mask(lambda x: [cv2.polylines(x, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, 1, 5) for q in it.gt], it.gray.shape)
    k = lambda n: np.ones((n, n), np.uint8)
    ring = cv2.dilate(m.astype(np.uint8), k(25)).astype(bool) & ~cv2.dilate(m.astype(np.uint8), k(11)).astype(bool)
    return float(abs(it.gray[m].mean() - it.gray[ring].mean())) if m.any() and ring.any() else np.nan


contrasts = np.asarray([lane_contrast(i) for i in items])
contrasts = contrasts[np.isfinite(contrasts)]
if not contrasts.size:
    raise SystemExit("Không đo được tương phản làn từ data/eval/normal.")
dI0 = float(np.median(contrasts))
if dI0 <= 0:
    raise SystemExit("Tương phản làn đo được bằng 0; không thể dự đoán điểm gãy.")
pred = 1 - args.k * args.noise / dI0
print(f"Dự đoán trước E4: điểm gãy d*={pred:.2f} (dI0={dI0:.1f}, nhiễu={args.noise}, k={args.k}).")

rows, curves = [], {}
for en, es in EDGES.items():
    for mn, fn in [("baseline", lambda g: run_baseline(g, P)), ("P", lambda g: run_proposed(g, P))]:
        f1s = []
        for d in DEPTHS:
            sub = [Item(i.tier, i.path, apply_shadow(i.gray, d, make_shadow(i.gray.shape, k, es), k), i.gt) for k, i in enumerate(items)]
            r = evaluate(fn, sub); f1s.append(r["f1"])
            rows.append(dict(method=mn, edge=en, depth=d, f1=round(r["f1"], 3), precision=round(r["precision"], 3), recall=round(r["recall"], 3)))
        curves[(mn, en)] = f1s
        brk = next((d for d, f in zip(DEPTHS, f1s) if f < 0.5 * f1s[0]), None)
        print(f"{mn} | biên {en} | F1={[round(f, 2) for f in f1s]} | điểm gãy={brk}.")
write_table("e4_break", rows)

plt.figure(figsize=(8, 5))
for (m, e), f in curves.items():
    plt.plot(DEPTHS, f, "o-", label=f"{m}, biên {e}")
plt.axvline(pred, color="k", ls="--", label=f"dự đoán d*={pred:.2f}")
plt.xlabel("độ sâu bóng d"); plt.ylabel("F1"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(figures_dir() / "e4_break.png", dpi=110)
