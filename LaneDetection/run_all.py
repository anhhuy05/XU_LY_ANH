import subprocess
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None)
a = ap.parse_args()
extra = ["--max", str(a.max)] if a.max else []
steps = [("tools.make_stratification", []), ("experiments.e2_params", extra), ("experiments.e1_challenge", extra),
         ("experiments.e3_ablation", extra), ("experiments.e4_breaking_point", extra)]
for mod, args in steps:
    print(f"\n[Chạy] {mod}", flush=True)
    subprocess.run([sys.executable, "-m", mod, *args], check=True)
