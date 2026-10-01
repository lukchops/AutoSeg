"""Baseline (drift, bend, wander, steps), detector noise, and artifacts -- all independent of the peaks."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .rand import loguniform, randint, uniform


def smooth_noise(rng, n, sigma_samples):
    """Gaussian-filtered white noise with unit std."""
    z = gaussian_filter1d(rng.normal(size=n), sigma_samples, mode="reflect")
    return z / z.std()


def baseline(rng, b, n, fs):
    """Return (baseline curve, params, step artifacts)."""
    x = np.linspace(-1, 1, n)
    offset = uniform(rng, b["offset_mau"])
    drift = uniform(rng, b["drift_mau"])  # total linear gain/loss over the run
    curve = uniform(rng, b["curve_mau"])
    c2, c3 = rng.uniform(-1, 1, 2)
    wander = uniform(rng, b["wander_mau"])
    wander_scale = uniform(rng, b["wander_scale_s"])
    y = (offset + drift * (x + 1) / 2 + curve * (c2 * x**2 + c3 * x**3)
         + wander * smooth_noise(rng, n, wander_scale * fs))

    s = b["steps"]
    t = np.arange(n) / fs
    steps = []
    for _ in range(randint(rng, s["count"])):
        t0 = rng.uniform(0, n / fs)
        mag = uniform(rng, s["mau"]) * rng.choice([-1, 1])
        rise = uniform(rng, s["rise_s"])
        y += mag * 0.5 * (1 + np.tanh(2 * (t - t0) / rise))
        steps.append({"type": "step", "start_idx": max(int((t0 - rise / 2) * fs), 0),
                      "end_idx": min(int((t0 + rise / 2) * fs), n - 1), "magnitude": float(mag)})

    params = {"offset_mau": offset, "drift_mau": drift, "curve_mau": curve,
              "curve_coefs": [float(c2), float(c3)], "wander_mau": wander, "wander_scale_s": wander_scale}
    return y, params, steps


def noise(rng, b, n, fs):
    """White Gaussian noise plus low-frequency (smoothed) noise. Returns (noise, params)."""
    sigma = loguniform(rng, b["noise_mau"])
    ratio = uniform(rng, b["lf_noise_ratio"])
    scale = uniform(rng, b["lf_noise_scale_s"])
    z = rng.normal(0, sigma, n) + ratio * sigma * smooth_noise(rng, n, scale * fs)
    return z, {"noise_mau": sigma, "lf_noise_ratio": ratio, "lf_noise_scale_s": scale}


def _regions(mask):
    """(start, end) index pairs (inclusive) of contiguous True runs."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1)


def apply_artifacts(rng, a, uv, fs):
    """Spikes, dropouts and clipping (applied last). Returns (uv, artifacts)."""
    uv = uv.copy()
    n = len(uv)
    arts = []

    s = a["spikes"]
    if rng.random() < s["prob"]:
        for _ in range(randint(rng, s["count"])):
            w = randint(rng, s["width_samples"])
            i = int(rng.integers(0, n - w))
            mag = uniform(rng, s["mau"]) * rng.choice([-1, 1])
            uv[i:i + w] += mag
            arts.append({"type": "spike", "start_idx": i, "end_idx": i + w - 1, "magnitude": float(mag)})

    d = a["dropouts"]
    if rng.random() < d["prob"]:
        for _ in range(randint(rng, d["count"])):
            w = max(int(uniform(rng, d["duration_s"]) * fs), 1)
            i = int(rng.integers(0, n - w))
            mode = "hold" if rng.random() < 0.5 else "zero"
            value = float(uv[i]) if mode == "hold" else 0.0
            uv[i:i + w] = value
            arts.append({"type": "dropout", "start_idx": i, "end_idx": i + w - 1,
                         "magnitude": value, "mode": mode})

    c = a["clipping"]
    if rng.random() < c["prob"]:
        med = np.median(uv)
        level = float(med + uniform(rng, c["frac"]) * (uv.max() - med))
        for i, j in _regions(uv >= level):
            arts.append({"type": "clip", "start_idx": int(i), "end_idx": int(j), "magnitude": level})
        uv = np.minimum(uv, level)

    return uv, arts
