import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
import config as cfg
from utils import roi_mask, postprocess_lines


def integral_image(a):
    a = np.asarray(a, np.float64)
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    S[1:, 1:] = a.cumsum(0).cumsum(1)
    return S


def local_mean(a, radius):
    w = 2 * radius + 1
    S = integral_image(np.pad(np.asarray(a, np.float64), radius, mode="edge"))
    return (S[w:, w:] - S[:-w, w:] - S[w:, :-w] + S[:-w, :-w]) / (w * w)


def illumination_norm(gray, normalize=True, sigma=1.5, radius=15):
    f = gaussian_filter(gray.astype(np.float64), sigma) if sigma > 0 else gray.astype(np.float64)
    if normalize:
        f = f / (local_mean(f, radius) + 1.0)
    return f


THR_NORM = (0.05, 0.12)
THR_RAW = (10.0, 30.0)


def sobel(f):
    p = np.pad(f, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def non_max_suppression(mag, gx, gy):
    H, W = mag.shape
    ang = np.arctan2(gy, gx) % np.pi
    sec = (((ang + np.pi / 8) // (np.pi / 4)) % 4).astype(int)
    mp = np.pad(mag, 1)
    sh = lambda dy, dx: mp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    c = [sec == 0, sec == 1, sec == 2]
    n1 = np.select(c, [sh(0, 1), sh(1, 1), sh(1, 0)], sh(1, -1))
    n2 = np.select(c, [sh(0, -1), sh(-1, -1), sh(-1, 0)], sh(-1, 1))
    return (mag >= n1) & (mag >= n2)


def hysteresis(nms, mag, lo, hi):
    lab, _ = ndi.label(nms & (mag >= lo), structure=np.ones((3, 3)))
    ok = np.unique(lab[nms & (mag >= hi)])
    return np.isin(lab, ok[ok > 0])


def canny_adaptive(f, roi, normalized=True, thr_scale=1.0):
    gx, gy = sobel(f)
    mag = np.hypot(gx, gy)
    lo, hi = THR_NORM if normalized else THR_RAW
    edges = hysteresis(non_max_suppression(mag, gx, gy), mag, lo * thr_scale, hi * thr_scale) & roi
    return edges, np.arctan2(gy, gx) % np.pi, mag


def hough_vote(edges, gdir, d_rho=2.0, d_theta=1.0, delta=10.0, vote=True):
    H, W = edges.shape
    n_th = int(round(180 / d_theta))
    dth = np.pi / n_th
    nh = int(np.ceil(np.hypot(H, W) / d_rho))
    n_rho = 2 * nh + 1
    ys, xs = np.nonzero(edges)
    A = np.zeros(n_rho * n_th)
    for s in range(0, len(xs), 20000):
        x = xs[s:s + 20000, None].astype(float)
        y = ys[s:s + 20000, None].astype(float)
        if vote:
            m = int(np.ceil(np.deg2rad(delta) / dth))
            tg = gdir[ys[s:s + 20000], xs[s:s + 20000]]
            idx = np.round(tg / dth).astype(int)[:, None] + np.arange(-m, m + 1)[None, :]
        else:
            idx = np.broadcast_to(np.arange(n_th), (x.shape[0], n_th))
        th = idx * dth
        rho = x * np.cos(th) + y * np.sin(th)
        rho = np.where((idx < 0) | (idx >= n_th), -rho, rho)
        idx = idx % n_th
        ri = np.round(rho / d_rho).astype(int) + nh
        ok = (ri >= 0) & (ri < n_rho)
        A += np.bincount(ri[ok] * n_th + idx[ok], minlength=n_rho * n_th)
    return A.reshape(n_rho, n_th)


def find_peaks(A, d_rho=2.0, d_theta=1.0, n_peaks=16, smooth=1.0, nms=6, rel=0.1, min_votes=6.0, min_horiz_deg=0.0):
    P = int(np.ceil(3 * smooth)) + nms
    Ap = np.concatenate([A[::-1, -P:], A, A[::-1, :P]], axis=1)
    S = ndi.gaussian_filter(Ap, smooth) if smooth > 0 else Ap
    ok = (S == ndi.maximum_filter(S, size=2 * nms + 1)) & (S >= max(min_votes, rel * S.max()))
    S, ok = S[:, P:-P], ok[:, P:-P]
    th = np.arange(A.shape[1]) * np.deg2rad(d_theta)
    ok = ok & (np.abs(th - np.pi / 2) >= np.deg2rad(min_horiz_deg))[None, :]
    ys, xs = np.nonzero(ok)
    nh = A.shape[0] // 2
    order = np.argsort(-S[ys, xs])[:n_peaks]
    return [((ys[i] - nh) * d_rho, xs[i] * np.deg2rad(d_theta), float(S[ys[i], xs[i]])) for i in order]


def hough_gradient(edges, gdir, vote=True, delta=10.0, smooth=1.0, d_rho=2.0, d_theta=1.0, min_horiz_deg=0.0, nms=6):
    A = hough_vote(edges, gdir, d_rho, d_theta, delta, vote)
    return find_peaks(A, d_rho, d_theta, smooth=smooth, nms=nms, min_horiz_deg=min_horiz_deg), A


def run_proposed(gray, p=None, debug=False):
    p = p or cfg.Params()
    roi = roi_mask(gray.shape, top=p.p_roi_top, x=(p.p_roi_x_left, p.p_roi_x_right))
    f = illumination_norm(gray, p.use_norm, p.sigma, p.radius)
    edges, gdir, mag = canny_adaptive(f, roi, p.use_norm, p.thr_scale)
    peaks, A = hough_gradient(edges, gdir, p.use_vote, p.delta, p.smooth,
                              min_horiz_deg=cfg.MIN_HORIZ_DEG, nms=p.peak_nms)
    lines = postprocess_lines(peaks, gray.shape)
    return (lines, dict(edges=edges, norm=f, mag=mag, A=A)) if debug else lines
