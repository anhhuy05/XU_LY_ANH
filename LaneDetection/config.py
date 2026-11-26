import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def resolve_data_path(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    resolved = candidate.resolve()
    data_root = DATA_ROOT.resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError:
        raise ValueError(f"Data paths must stay inside {data_root}: {resolved}") from None
    return resolved


DATA = resolve_data_path(os.environ.get("LANE_DATA") or DATA_ROOT)
RESULTS = ROOT / "results"


def _tiers():
    d = DATA / "eval"
    found = [x.name for x in d.iterdir() if x.is_dir()] if d.exists() else []
    for order in (["normal", "hlight", "shadow"], ["light", "medium", "heavy"]):
        if found and set(found) <= set(order):
            return [t for t in order if t in found]
    return sorted(found) or ["normal", "hlight", "shadow"]


TIERS = _tiers()

SCALE = 0.5
ROI_TOP = 0.45
ROI_BOTTOM = 1.0
ROI_X = (0.30, 0.70)
LANE_W = 30
IOU_THRESH = 0.5
MAX_LANES = 4
MIN_HORIZ_DEG = 25
DEDUP = 0.04


@dataclass
class Params:

    b_lo: int = 50
    b_hi: int = 150
    b_thresh: int = 40

    use_norm: bool = True
    use_vote: bool = True
    sigma: float = 1.5
    radius: int = 15
    thr_scale: float = 1.0
    delta: float = 10.0
    smooth: float = 1.0
    peak_nms: int = 3
    p_roi_top: float = 0.45
    p_roi_x_left: float = 0.3
    p_roi_x_right: float = 0.7
