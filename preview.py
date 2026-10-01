"""Plot chromatograms with their peak table and artifacts overlaid.

    py -3.12 preview.py data/realistic_v1              # saved files (first --n, from --start)
    py -3.12 preview.py --preset hard --seed 3         # generate on the fly, nothing saved
    py -3.12 preview.py --families --preset realistic  # the unit templates of every shape family

Shading = peak [start, end] (color = group, hatched = negative dip), number = peak id.
Red x = spike, orange dashed = step; top-edge strip: black = dropout, red = clipped.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from peakmaker.compose import generate_one
from peakmaker.config import load_preset
from peakmaker.io import load_chromatogram
from peakmaker.real_templates import load_real_templates
from peakmaker.shapes import SHAPES, U
from peakmaker.warp import shape_noise

ART_STYLE = {"dropout": "k", "clip": "tab:red"}


def plot_chromatogram(ax, uv, fs, meta, title):
    t = np.arange(len(uv)) / fs / 60
    ax.plot(t, uv, lw=0.7, color="k")
    cmap = plt.get_cmap("tab10")
    for p in meta["peaks"]:
        c = cmap(p["group_id"] % 10)
        dip = p["polarity"] < 0
        ax.axvspan(t[p["start_idx"]], t[p["end_idx"]], color=c, alpha=0.2, hatch="//" if dip else None)
        i = p["apex_idx"]
        ax.annotate(str(p["id"]), (t[i], uv[i]), textcoords="offset points", xytext=(0, -12 if dip else 4),
                    ha="center", fontsize=7, color=c)
    for a in meta["artifacts"]:
        i, j = a["start_idx"], a["end_idx"]
        if a["type"] == "spike":
            ax.plot(t[i:j + 1], uv[i:j + 1], "x", color="tab:red", ms=5)
        elif a["type"] == "step":
            ax.axvline(t[(i + j) // 2], color="tab:orange", ls="--", lw=0.8)
        else:  # drawn as a strip along the top edge so it can't be confused with peak shading
            ax.axvspan(t[i], t[j], ymin=0.92, ymax=1.0, color=ART_STYLE[a["type"]])
    bg = meta["background"]
    ax.set_title(f"{title}: {len(meta['peaks'])} peaks, {len(meta['artifacts'])} artifacts, "
                 f"noise {bg['noise_mau']:.2f} mAU, {len(uv) / fs / 60:.1f} min", fontsize=9)
    ax.set_ylabel("UV (mAU)")


def plot_families(cfg, real, seed):
    rng = np.random.default_rng(seed)
    shapes = {f: p for f, p in cfg["families"].items() if f in SHAPES}
    names = list(shapes) + (["real"] if real else [])
    fig, axes = plt.subplots(3, 3 if len(names) <= 9 else 4, figsize=(14, 9), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for _ in range(5):
            if name == "real":
                tpl = real[rng.integers(len(real))][1]
            else:
                tpl = SHAPES[name](rng, shapes[name])[0]
            ax.plot(U, shape_noise(rng, tpl, cfg["peaks"]["shape_noise"]), lw=0.8)
        ax.set_xlim(-5, 10)
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("unit templates (u in FWHM units), 5 random draws each")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="dataset folder with chrom_*.json / .npz")
    ap.add_argument("--preset", default="realistic", help="preset for on-the-fly generation / --families")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=6, help="number of chromatograms to show")
    ap.add_argument("--start", type=int, default=0, help="first file index when reading a folder")
    ap.add_argument("--families", action="store_true", help="plot unit templates of every shape family")
    ap.add_argument("--save", help="save the figure to this png instead of showing it")
    args = ap.parse_args()

    if args.families or not args.path:
        cfg = load_preset(args.preset)
        real = load_real_templates(cfg["real_templates"])

    if args.families:
        plot_families(cfg, real, args.seed)
    else:
        if args.path:
            all_files = sorted(Path(args.path).glob("chrom_*.json"))
            files = all_files[args.start:args.start + args.n]
            if not files:
                raise SystemExit(f"no chromatograms to show: {args.path} has {len(all_files)} "
                                 f"(chrom_00000..), --start is {args.start}")
            items =[(f.stem, *load_chromatogram(f)) for f in files]
        else:
            items = []
            for i in range(args.n):
                r = generate_one(cfg, real, args.seed, i)
                items.append((f"{Path(args.preset).stem} seed {args.seed} #{i}", r["uv"], r["fs"], r))
        fig, axes = plt.subplots(len(items), 1, figsize=(14, 2.4 * len(items)), squeeze=False)
        for ax, (title, uv, fs, meta) in zip(axes[:, 0], items):
            plot_chromatogram(ax, uv, fs, meta, title)
        axes[-1, 0].set_xlabel("time (min)")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()
