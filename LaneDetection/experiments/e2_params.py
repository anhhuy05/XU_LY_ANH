import sys, argparse
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
from utils import load_items, evaluate, evaluate_by_tier, write_table, figures_dir, illum_stats, save_params, overlay, video_id
from baseline import run_baseline
from proposed import run_proposed

ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None)
args = ap.parse_args()
dev = [it._replace(tier=it.path.parent.name) for it in load_items("dev", args.max)]
dev_ratios = [illum_stats(it.gray)[1] for it in dev]
contrast_cut = float(np.quantile(dev_ratios, 0.75))
contrast_items = [it for it, ratio in zip(dev, dev_ratios) if ratio >= contrast_cut]
contrast_by_video = {}
for it in contrast_items:
    contrast_by_video.setdefault(video_id(it.path), []).append(it)


def dev_metrics(fn):
    by_tier = evaluate_by_tier(fn, dev)
    tier_f1 = {tier: result["f1"] for tier, result in by_tier.items()}
    contrast_video_f1 = float(np.mean([
        evaluate(fn, items)["f1"] for items in contrast_by_video.values()
    ]))
    other_f1 = [score for tier, score in tier_f1.items() if tier != "shadow"]
    if "shadow" in tier_f1 and other_f1:
        score = 0.5 * tier_f1["shadow"] + 0.25 * contrast_video_f1 + 0.25 * float(np.mean(other_f1))
    else:
        score = 0.5 * contrast_video_f1 + 0.5 * float(np.mean(list(tier_f1.values())))
    metrics = {
        key: float(np.mean([result[key] for result in by_tier.values()]))
        for key in ("f1", "precision", "recall", "ms")
    }
    metrics["score"] = score
    metrics["shadow_f1"] = float(tier_f1.get("shadow", float("nan")))
    metrics["contrast_video_f1"] = contrast_video_f1
    return metrics


rows = []
bf = -1
for lo, hi in [(30, 90), (50, 150), (80, 200)]:
    for th in [25, 40, 60]:
        c = replace(cfg.Params(), b_lo=lo, b_hi=hi, b_thresh=th)
        r = dev_metrics(lambda g: run_baseline(g, c))
        rows.append(dict(param="baseline(lo/hi/thresh)", value=f"{lo}/{hi}/{th}",
                         score=round(r["score"], 4), f1=round(r["f1"], 4),
                         shadow_f1=round(r["shadow_f1"], 4),
                         contrast_video_f1=round(r["contrast_video_f1"], 4),
                         eligible="",
                         precision=round(r["precision"], 4), recall=round(r["recall"], 4),
                         ms=round(r["ms"], 1)))
        if r["score"] > bf:
            bf, bb = r["score"], (lo, hi, th)
baseline = replace(cfg.Params(), b_lo=bb[0], b_hi=bb[1], b_thresh=bb[2])
baseline_shadow_f1 = dev_metrics(lambda g: run_baseline(g, baseline))["shadow_f1"]

SWEEPS = [("thr_scale", [0.35, 0.4, 0.45, 0.5, 0.75, 1.0, 1.5, 2.0]), ("radius", [5, 10, 15, 25, 40]),
      ("delta", [3.0, 6.0, 10.0, 15.0, 25.0]), ("smooth", [0.0, 0.5, 1.0, 2.0]),
      ("peak_nms", [3, 4, 6, 8]), ("p_roi_top", [0.45, 0.5, 0.55, 0.6]),
      ("p_roi_x_left", [0.25, 0.3, 0.35, 0.4]), ("p_roi_x_right", [0.6, 0.65, 0.7, 0.75])]
best = cfg.Params(b_lo=baseline.b_lo, b_hi=baseline.b_hi, b_thresh=baseline.b_thresh,
                  thr_scale=0.45, radius=15, delta=6.0, smooth=0.0, peak_nms=3,
                  p_roi_top=0.55, p_roi_x_left=0.35, p_roi_x_right=0.6)
curves = {}
for name, vals in SWEEPS:
    scores = []
    for v in vals:
        r = dev_metrics(lambda g: run_proposed(g, replace(best, **{name: v})))
        eligible = r["shadow_f1"] >= baseline_shadow_f1
        scores.append(r["score"] if eligible else float("-inf"))
        rows.append(dict(param=name, value=v, score=round(r["score"], 4), f1=round(r["f1"], 4),
                         shadow_f1=round(r["shadow_f1"], 4), contrast_video_f1=round(r["contrast_video_f1"], 4),
                         eligible=eligible, precision=round(r["precision"], 4),
                         recall=round(r["recall"], 4), ms=round(r["ms"], 1)))
        status = "đạt" if eligible else "loại"
        print(f"{name}={v}: điểm={r['score']:.4f}, shadow={r['shadow_f1']:.4f} ({status}).")
    best = replace(best, **{name: vals[int(np.argmax(scores))]})
    curves[name] = (vals, scores)

refine_scales = [0.35, 0.4, 0.45, 0.5, 0.75]
refine_scores = []
for scale in refine_scales:
    r = dev_metrics(lambda g, scale=scale: run_proposed(g, replace(best, thr_scale=scale)))
    eligible = r["shadow_f1"] >= baseline_shadow_f1
    refine_scores.append(r["score"] if eligible else float("-inf"))
    rows.append(dict(param="thr_scale_refine", value=scale, score=round(r["score"], 4),
                     f1=round(r["f1"], 4), shadow_f1=round(r["shadow_f1"], 4),
                     contrast_video_f1=round(r["contrast_video_f1"], 4), eligible=eligible,
                     precision=round(r["precision"], 4),
                     recall=round(r["recall"], 4), ms=round(r["ms"], 1)))
    status = "đạt" if eligible else "loại"
    print(f"thr_scale={scale}: điểm={r['score']:.4f}, shadow={r['shadow_f1']:.4f} ({status}).")
best = replace(best, thr_scale=refine_scales[int(np.argmax(refine_scores))])
curves["thr_scale"] = (refine_scales, refine_scores)

selected_metrics = dev_metrics(lambda g: run_proposed(g, best))
print(f"Dev shadow F1 | baseline={baseline_shadow_f1:.4f}, P={selected_metrics['shadow_f1']:.4f}.")
print(f"Dev score P={selected_metrics['score']:.4f}; tham số: {best}.")
save_params(best)
write_table("e2_sweep", rows)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (n, (v, f)) in zip(ax, curves.items()):
    a.plot(v, f, "o-"); a.set_title(n); a.set_ylabel("Shadow-robust score (dev)"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(figures_dir() / "e2_curves.png", dpi=110); plt.close()


it = max(dev, key=lambda i: illum_stats(i.gray)[1])
lb, db = run_baseline(it.gray, best, True)
lp, dp = run_proposed(it.gray, best, True)
_, dn = run_proposed(it.gray, replace(best, use_norm=False), True)
_, dv = run_proposed(it.gray, replace(best, use_vote=False), True)
panels = [("ảnh xám", it.gray), ("sau chuẩn hóa chiếu sáng", dp["norm"]), ("|gradient| chuẩn hóa", dp["mag"]),
          ("biên baseline", db["edges"]), ("biên P (KHÔNG chuẩn hóa)", dn["edges"]), ("biên P", dp["edges"]),
          ("A: bỏ phiếu mọi theta", dv["A"]), ("A: bỏ phiếu theo gradient", dp["A"]),
          ("kết quả baseline", overlay(it, lb)[..., ::-1]), ("kết quả P", overlay(it, lp)[..., ::-1])]
fig, ax = plt.subplots(2, 5, figsize=(24, 8))
for a, (t, im) in zip(ax.ravel(), panels):
    a.imshow(im, cmap="gray", aspect="auto"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(figures_dir() / "e2_intermediate.png", dpi=100); plt.close()
