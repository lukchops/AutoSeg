"""Assemble one labeled synthetic chromatogram: peaks on top of an independent background."""
import numpy as np

from .background import apply_artifacts, baseline, noise
from .labels import build_table, visibility
from .rand import loguniform, randint, uniform
from .shapes import COMPOSITES, SHAPES
from .warp import place, shape_noise


def _pick_family(rng, families, exclude=()):
    names = [f for f, p in families.items() if p["weight"] > 0 and f not in exclude]
    if not names:
        raise ValueError("no usable shape family (all weights 0, or only 'real' enabled without templates)")
    w =np.array([families[f]["weight"] for f in names], float)
    return names[rng.choice(len(names), p=w / w.sum())]


def _template(rng, cfg, family, real):
    """Draw one unit template -> (template, source, shape params)."""
    if family == "real":
        name, tpl = real[rng.integers(len(real))]
        return tpl, f"real:{name}", {}
    tpl, sp = SHAPES[family](rng, cfg["families"][family])
    return tpl, "parametric", sp


def _local_noise(bg, unit, frac):
    """Std of the background under a peak's footprint after removing a straight line.

    Noise, wander, curvature and steps under the peak all count; a plain slope does not,
    since a peak sitting on a slope is still easy to see.
    """
    idx = np.nonzero(unit >= frac * unit.max())[0]
    s, e = idx[0], idx[-1]
    if e - s < 4:
        c = (s + e) // 2
        s, e = max(c - 2, 0), min(c + 2, len(bg) - 1)
    seg = bg[s:e + 1]
    x = np.arange(len(seg))
    return float(np.std(seg - np.polyval(np.polyfit(x, seg, 1), x)))


def generate_chromatogram(rng, cfg, real):
    fs = cfg["fs"]
    pc = cfg["peaks"]
    fams = cfg["families"] if real else {k: v for k, v in cfg["families"].items() if k != "real"}
    n = int(round(uniform(rng, cfg["length_min"]) * 60 * fs))
    dur = n / fs
    n_target = randint(rng, cfg["peak_count"])

    # background first, so every peak can be sized against the background under it
    base, bg_params, steps = baseline(rng, cfg["background"], n, fs)
    z, noise_params = noise(rng, cfg["background"], n, fs)
    bg = base + z

    comps, infos, group = [], [], 0
    while len(infos) < n_target:
        # composites add two peaks; don't overshoot the drawn peak count
        family = _pick_family(rng, fams, COMPOSITES if n_target - len(infos) < 2 else ())
        fp = fams[family]
        fwhm = loguniform(rng, fp.get("fwhm_s", pc["fwhm_s"]))
        polarity = -1 if rng.random() < pc["negative_prob"] else 1
        apex = rng.uniform(pc["edge_margin_s"], dur - pc["edge_margin_s"])

        if family in COMPOSITES:
            # sub-peak offset from resolution Rs = dt / (0.5 * (w1 + w2)), baseline width w ~= 1.7 * FWHM
            delta = uniform(rng, fp["rs"]) * 1.7 * fwhm * rng.choice([-1, 1])
            subs = [(str(rng.choice(fp["sub_families"])), apex, fwhm, 1.0),
                    (str(rng.choice(fp["sub_families"])), float(np.clip(apex + delta, 0, dur)),
                     fwhm * rng.uniform(0.8, 1.25), uniform(rng, fp["height_ratio"]))]
        else:
            subs = [(family, apex, fwhm, 1.0)]

        for k, (shape, a, w, ratio) in enumerate(subs):
            tpl, source, sp = _template(rng, cfg, shape, real)
            asym = float(np.exp(rng.normal(0, pc["asym_sigma"])))
            tpl = shape_noise(rng, tpl, pc["shape_noise"])
            unit = place(tpl, n, fs, a, w, asym)
            local = _local_noise(bg, unit, cfg["boundary_frac"])
            floor = pc["min_snr"] * local
            if k == 0:  # main peak: log-uniform height, but never below the detectability floor
                lo, hi = pc["height_mau"]
                height = h = loguniform(rng, [min(max(lo, floor), hi), hi])
            else:       # composite sub-peak: fraction of the main peak, same floor
                h = max(ratio * height, floor)
            comps.append(polarity * h * unit)
            infos.append({"family": family, "shape": shape, "source": source, "group_id": group,
                          "polarity": polarity, "snr": h / local, "shape_params": sp,
                          "warp": {"fwhm_s_target": w, "asym": asym, "shape_noise": pc["shape_noise"]}})
        group += 1

    # drop buried peaks (no feature of their own in the summed signal), least visible first,
    # re-scoring after each removal since the sum changes
    n_buried = 0
    while comps:
        vis = visibility(comps)
        worst = int(np.argmin(vis))
        if vis[worst] >= pc["min_visibility"]:
            break
        del comps[worst], infos[worst]
        n_buried += 1
    for info, v in zip(infos, vis if comps else []):
        info["visibility"] = v

    peaks = build_table(comps, infos, fs, cfg["boundary_frac"])
    uv, arts = apply_artifacts(rng, cfg["artifacts"], bg + sum(comps, np.zeros(n)), fs)

    return {"uv": uv.astype(np.float32), "fs": fs, "length": n, "buried_removed": n_buried, "peaks": peaks,
            "artifacts": sorted(steps + arts, key=lambda x: x["start_idx"]),
            "background": {**bg_params, **noise_params}}


def generate_one(cfg, real, master_seed, index):
    """Chromatogram `index` of the dataset seeded by `master_seed` (same as SeedSequence(master_seed).spawn(n)[index])."""
    rng = np.random.default_rng(np.random.SeedSequence(master_seed, spawn_key=(index,)))
    return {"seed": {"master": master_seed, "index": index}, **generate_chromatogram(rng, cfg, real)}
