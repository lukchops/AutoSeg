"""Interactive picker: browse annotated PRIMAS peaks and save a curated handful as real unit templates.

    py -3.12 pick_templates.py                      # default PRIMAS 2.0 + 2.1 annotated folders
    py -3.12 pick_templates.py "path/to/annotated" --start 200

Keys:  a = accept   n = next (skip)   b = back   q = save and quit
Accepted peaks are appended to templates/real_templates.npz (existing templates are kept).
The source CSVs are only read, never modified.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from peakmaker.config import ROOT
from peakmaker.shapes import U, normalize

DEFAULT_DIRS = [ROOT.parent / "BIASEP/IVT/PRIMAS 2.0/Raw data/annotated",
                ROOT.parent / "BIASEP/IVT/PRIMAS 2.1/RT/Raw data/annotated"]


def iter_peaks(folders):
    """Annotated CSVs: 4-column blocks (Time(min), <s>.Absorbance..., <s>.Conductivity..., peak_id)."""
    for f in sorted(p for d in folders for p in Path(d).glob("*.csv")):
        df = pd.read_csv(f)
        for b in range(0, df.shape[1], 4):
            uv = df.iloc[:, b + 1]
            if uv.isna().all():
                continue
            t = df.iloc[:, b].to_numpy(float) * 60
            uv = uv.interpolate().bfill().ffill().to_numpy(float)
            pid = df.iloc[:, b + 3].fillna(0).to_numpy(int)
            sample = df.columns[b + 1].split(".Absorbance")[0]
            for p in sorted(set(pid[pid > 0])):
                idx = np.nonzero(pid == p)[0]
                yield {"name": f"{f.stem}|{sample}|p{p}", "t": t, "uv": uv, "start": idx[0], "end": idx[-1]}


def extract(pk, margin):
    """Baseline-subtracted, PCHIP-upsampled unit template of one annotated peak (None if not extractable)."""
    s, e = max(pk["start"] - margin, 0), min(pk["end"] + margin, len(pk["uv"]) - 1)
    x, y = pk["t"][s:e + 1], pk["uv"][s:e + 1]
    y = y - np.interp(x, [x[0], x[-1]], [y[:2].mean(), y[-2:].mean()])
    if len(x) < 4 or y.max() <= 0:
        return None
    # no undershoot below the fitted baseline, and ramp the window edges to 0 so the template has no steps
    y = np.maximum(y, 0)
    ramp = min(margin, len(y) // 4)
    if ramp > 0:
        taper = np.linspace(0, 1, ramp + 1)[:-1]
        y[:ramp] *= taper
        y[-ramp:] *= taper[::-1]
    xf = np.linspace(x[0], x[-1], 50 * len(x))
    try:
        return normalize(xf, PchipInterpolator(x, y)(xf))
    except IndexError:  # does not drop below half height inside the window
        return None


class Picker:
    def __init__(self, peaks, start, margin, names):
        self.peaks, self.i, self.margin = peaks, start, margin
        self.names = set(names)
        self.accepted = []
        self.fig, (self.ax_run, self.ax_tpl) = plt.subplots(
            1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def draw(self):
        pk = self.peaks[self.i]
        self.tpl = extract(pk, self.margin)
        self.ax_run.clear()
        self.ax_tpl.clear()
        t = pk["t"] / 60
        self.ax_run.plot(t, pk["uv"], lw=0.8, color="k")
        self.ax_run.axvspan(t[pk["start"]], t[pk["end"]], color="tab:blue", alpha=0.25)
        self.ax_run.set_title(f"[{self.i + 1}/{len(self.peaks)}] {pk['name']}   accepted this session: "
                              f"{len(self.accepted)}   (a accept, n next, b back, q save+quit)", fontsize=9)
        self.ax_run.set_xlabel("time (min)")
        if self.tpl is None:
            self.ax_tpl.text(0.5, 0.5, "cannot extract\n(no half-height crossing)", ha="center",
                             transform=self.ax_tpl.transAxes)
        else:
            self.ax_tpl.plot(U, self.tpl)
            self.ax_tpl.set_xlim(-5, 10)
            dup = " (already saved)" if pk["name"] in self.names else ""
            self.ax_tpl.set_title(f"unit template{dup}", fontsize=9)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "a" and self.tpl is not None and self.peaks[self.i]["name"] not in self.names:
            self.accepted.append((self.peaks[self.i]["name"], self.tpl))
            self.names.add(self.peaks[self.i]["name"])
        if event.key in ("a", "n"):
            self.i = min(self.i + 1, len(self.peaks) - 1)
        elif event.key == "b":
            self.i = max(self.i - 1, 0)
        elif event.key == "q":
            plt.close(self.fig)
            return
        self.draw()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", default=DEFAULT_DIRS, help="folders with annotated CSVs")
    ap.add_argument("--out", default=str(ROOT / "templates" / "real_templates.npz"))
    ap.add_argument("--start", type=int, default=0, help="index of the first peak to show")
    ap.add_argument("--margin", type=int, default=3, help="extra samples around the annotated [start, end]")
    args = ap.parse_args()

    peaks = list(iter_peaks(args.folders))
    print(f"{len(peaks)} annotated peaks")
    names, tpls = [], []
    if Path(args.out).exists():
        d = np.load(args.out)
        names = [str(n) for n in d["names"]]
        tpls = [np.interp(U, d["u"], t, left=0.0, right=0.0) for t in d["templates"]]
        print(f"{len(names)} existing templates in {args.out}")

    picker = Picker(peaks, args.start, args.margin, names)
    plt.show()

    if picker.accepted:
        names += [n for n, _ in picker.accepted]
        tpls += [t for _, t in picker.accepted]
        np.savez(args.out, u=U, templates=np.stack(tpls), names=np.array(names))
        print(f"saved {len(names)} templates ({len(picker.accepted)} new) to {args.out}")
    else:
        print("nothing accepted, file unchanged")


if __name__ == "__main__":
    main()
