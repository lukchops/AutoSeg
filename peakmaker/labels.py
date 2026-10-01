"""Peak table: measure each clean (noise-free) component and link overlapping peaks."""
import numpy as np


def measure(comp, fs, frac):
    """Apex, signed height, [start, end] at `frac` of own height, interpolated FWHM, area."""
    a = np.abs(comp)
    apex = int(np.argmax(a))
    h = float(comp[apex])
    above = np.nonzero(a >= frac * abs(h))[0]
    half = np.nonzero(a >= 0.5 * abs(h))[0]
    l, r = half[0], half[-1]
    hh = 0.5 * abs(h)
    left = l - (a[l] - hh) / (a[l] - a[l - 1]) if l > 0 else l
    right = r + (a[r] - hh) / (a[r] - a[r + 1]) if r < len(a) - 1 else r
    return {
        "apex_idx": apex,
        "start_idx": int(above[0]),
        "end_idx": int(above[-1]),
        "height": h,
        "fwhm_s": float((right - left) / fs),
        "area": float(comp.sum() / fs),
        "truncated": bool(a[0] >= frac * abs(h) or a[-1] >= frac * abs(h)),
    }


def visibility(comps):
    """Per peak, 0..1: how clearly the summed clean signal shows a feature of its own.

    A feature is a local minimum of |slope| inside the peak's half-height window -- its own maximum,
    or a shoulder where the slope flattens -- that is nearer to this peak's apex than to any other
    apex (ties go to the taller peak). A valley between peaks doesn't count. Score = how much the
    slope flattens there relative to the steepest slope within one window length on each side
    (1 = a true local maximum, 0 = no feature: the peak just makes a neighbour taller/wider).
    """
    c = np.sum(comps, axis=0)
    apex = np.array([int(np.argmax(np.abs(k))) for k in comps])
    h = np.array([abs(k[a]) for k, a in zip(comps, apex)])
    scores = []
    for i, k in enumerate(comps):
        d1 = np.gradient(np.sign(k[apex[i]]) * c)
        a = np.abs(d1)
        w = np.nonzero(np.abs(k) >= 0.5 * h[i])[0]
        span = max(w[-1] - w[0] + 1, 3)
        j = np.arange(max(w[0], 1), min(w[-1], len(c) - 2) + 1)
        j = j[(a[j] <= a[j - 1]) & (a[j] < a[j + 1]) & ~((d1[j - 1] < 0) & (d1[j + 1] > 0))]
        owner = np.argmin(np.abs(j[:, None] - apex[None, :]) - 1e-6 * h[None, :] / h.max(), axis=1)
        best = 0.0
        for jj in j[owner == i]:
            ref = min(a[max(jj - span, 0):jj + 1].max(), a[jj:jj + span + 1].max())
            if ref > 0:
                best = max(best, 1 - a[jj] / ref)
        scores.append(float(best))
    return scores


def build_table(comps, infos, fs, frac):
    """One row per component, sorted by apex; `id` = order of appearance (1-based)."""
    rows = [{**measure(c, fs, frac), **info} for c, info in zip(comps, infos)]
    rows.sort(key=lambda r: r["apex_idx"])
    for i, r in enumerate(rows):
        r["id"] = i + 1
    for r in rows:
        r["overlaps_with"] = [o["id"] for o in rows if o is not r
                              and o["start_idx"] <= r["end_idx"] and r["start_idx"] <= o["end_idx"]]
    return [{"id": r.pop("id"), **r} for r in rows]
