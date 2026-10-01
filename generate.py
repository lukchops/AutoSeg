"""Generate a synthetic chromatogram dataset.

    py -3.12 generate.py --preset realistic --n 1000 --seed 42 --out data/realistic_v1
"""
import argparse
from pathlib import Path

from peakmaker.compose import generate_one
from peakmaker.config import load_preset
from peakmaker.io import save_chromatogram, save_manifest
from peakmaker.real_templates import load_real_templates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="realistic", help="preset name in presets/ or path to a .yaml")
    ap.add_argument("--n", type=int, required=True, help="number of chromatograms")
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    cfg = load_preset(args.preset)
    real = load_real_templates(cfg["real_templates"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(out, args.preset, cfg, args.seed, args.n)
    for i in range(args.n):
        save_chromatogram(out, i, generate_one(cfg, real, args.seed, i))
        if (i + 1) % 100 == 0 or i + 1 == args.n:
            print(f"{i + 1}/{args.n}")


if __name__ == "__main__":
    main()
