import sys, argparse
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
from utils import load_items, evaluate, evaluate_by_tier, write_table, figures_dir, load_params
from baseline import run_baseline
from proposed import run_proposed

ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None); ap.add_argument("--dev-max", type=int, default=None)
args = ap.parse_args()
P = load_params()
items = load_items("eval", args.max)


dev = load_items("dev", args.dev_max)
sc = max([0.5, 1, 2, 4], key=lambda s: evaluate(lambda g: run_proposed(g, replace(P, use_norm=False, thr_scale=s)), dev)["f1"])
CONFIGS = {"P đầy đủ": lambda g: run_proposed(g, P),
           "P - chuẩn hóa (use_norm=False)": lambda g: run_proposed(g, replace(P, use_norm=False, thr_scale=sc)),
           "P - bỏ phiếu gradient": lambda g: run_proposed(g, replace(P, use_vote=False)),
           "baseline": lambda g: run_baseline(g, P)}
rows, table = [], {}
for name, fn in CONFIGS.items():
    table[name] = evaluate_by_tier(fn, items)
    for t, r in table[name].items():
        rows.append(dict(config=name, tier=t, n=r["n"], f1=round(r["f1"], 3), precision=round(r["precision"], 3),
                         recall=round(r["recall"], 3), lat_px=round(r["lat_px"], 1), ms=round(r["ms"], 1)))
write_table("e3_ablation", rows)

tiers = list(next(iter(table.values())))
fig, ax = plt.subplots(figsize=(10, 4.5)); w = 0.8 / len(tiers)
for k, t in enumerate(tiers):
    ax.bar([i + k * w for i in range(len(CONFIGS))], [table[c][t]["f1"] for c in CONFIGS], w, label=t)
ax.set_xticks([i + 0.4 - w / 2 for i in range(len(CONFIGS))]); ax.set_xticklabels(list(CONFIGS), rotation=20, ha="right")
ax.set_ylabel("F1"); ax.legend(); plt.tight_layout(); plt.savefig(figures_dir() / "e3_ablation.png", dpi=110)
print("E3 xong: so sánh F1 của P khi bật/tắt chuẩn hóa.")
