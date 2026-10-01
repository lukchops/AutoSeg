# Peak_maker

A synthetic UV chromatogram generator. It builds each chromatogram from a library of peak shapes: parametric ones plus a curated set of real PRIMAS peaks. Each shape is randomly warped and summed onto a baseline that is simulated separately (drift, steps, noise, artifacts), giving an effectively unlimited supply of labeled chromatograms for model training.

```
py -3.12 -m pip install -r requirements.txt
```

## Workflow

1. **Pick real templates** (optional). Without them, the `real` family is disabled with a warning.
   ```
   py -3.12 pick_templates.py
   ```
   This browses every annotated peak in the PRIMAS 2.0 and 2.1 annotated CSVs. The left panel shows the run and the right panel shows the unit template that would be saved. Keys: `a` accept, `n` skip, `b` back, `q` save and quit. Accepted peaks are added to `templates/real_templates.npz`. Run it again to add more; `--start N` jumps to peak N.
2. **Look at the shapes and a few runs**:
   ```
   py -3.12 preview.py --families --preset realistic
   py -3.12 preview.py --preset realistic --seed 3
   ```
3. **Generate a dataset**:
   ```
   py -3.12 generate.py --preset realistic --n 10000 --seed 42 --out data/realistic_v1
   py -3.12 preview.py data/realistic_v1 --start 20     # shows chrom_00020 .. chrom_00025
   ```

The same `--preset` and `--seed` always produce the same dataset.

## Output format

| file | content |
|---|---|
| `chrom_XXXXX.npz` | `uv` (float32, mAU) and `fs` (Hz); time = `arange(len(uv)) / fs` |
| `chrom_XXXXX.json` | `seed`, `length`, `buried_removed` (how many buried peaks were dropped), `peaks`, `artifacts`, `background` params |
| `manifest.json` | preset, master seed, n, SHA-256 of the real templates file, full resolved config |

**`peaks`**: one row per peak, sorted by apex. `id` is the order of appearance, starting at 1.

| key | meaning |
|---|---|
| `apex_idx`, `start_idx`, `end_idx` | sample indices; start/end are where the clean (noise-free) peak reaches `boundary_frac` (1%) of its own height |
| `height`, `fwhm_s`, `area` | measured on the clean peak; `height` is signed (negative for dips) |
| `polarity` | `1` for a peak, `-1` for a negative dip |
| `snr` | peak height / std of the background under the peak (after removing a straight line); always ≥ the preset's `min_snr` |
| `visibility` | 0–1, how clearly the peak shows its own maximum or shoulder in the summed signal; always ≥ `min_visibility` |
| `family` / `shape` / `source` | what was sampled: `family` is `shoulder`/`doublet` for composite sub-peaks, `shape` is the actual shape, `source` is `parametric` or `real:<file>\|<sample>\|p<id>` |
| `group_id` | shared by the sub-peaks of one composite (shoulder/doublet) |
| `overlaps_with` | ids of peaks whose [start, end] intersects this one |
| `truncated` | the peak is cut off by the run start or end |
| `shape_params`, `warp` | the random parameters used |

**`artifacts`**: `{type, start_idx, end_idx, magnitude}` with type `step`, `spike`, `dropout` (plus `mode`: `hold`/`zero`) or `clip`. Artifacts are never in the peak table.

## How a chromatogram is built

- Every shape is a *unit template* on a normalized axis: apex at u=0, height 1, FWHM 1 (`peakmaker/shapes.py`). Real templates are baseline-subtracted, PCHIP-upsampled and normalized the same way.
- Each placement goes through these steps (`peakmaker/warp.py`, `compose.py`):
  - smooth shape noise
  - left/right asymmetry stretch
  - resampling to a random FWHM and apex position at 10 Hz
  - a random signed height
- Composites add two sub-peaks, spaced from a resolution `Rs` range; each sub-peak gets its own row.
- The background (`peakmaker/background.py`) is drawn independently of the peaks: offset, linear gain/loss, bend and slow wander, smoothed steps, white and low-frequency noise, then spikes and dropouts. Clipping is applied last.

## Presets

`presets/easy.yaml`, `realistic.yaml` and `hard.yaml` hold every setting; see the comments in `realistic.yaml`. `[lo, hi]` means a random range. Heights, widths and noise are drawn log-uniformly; everything else is uniform. To make a new preset, copy a file and pass its path: `--preset my.yaml`.

Notes:
- `peak_count` counts peaks (table rows), so a composite uses up 2 of them.
- `min_snr` (easy 10, realistic 5, hard 3) removes undetectable peaks. For each peak, the generator measures the background under the peak's footprint: noise, wander, bends and steps count, a plain slope doesn't. No peak is made smaller than `min_snr` × that level. Main peaks keep a log-uniform height above the floor; composite sub-peaks are raised to it if their height ratio would put them below.
- `min_visibility` (default 0.3) removes peaks buried under neighbours. A peak survives only if the summed noise-free signal shows a feature of its own near its apex: its own maximum, or a shoulder where the slope clearly flattens. That feature must be nearer to its apex than to any other apex. A small peak on top of or on the flank of a big one, which only makes the big one taller or wider, is dropped; so is one of two equal peaks merged into a single blob. Least visible peaks go first, one at a time, so a run can end up with fewer peaks than `peak_count` drew. Two same-width peaks need Rs ≳ 0.6 for a small one (ratio 0.3) to show; the shoulder/doublet `rs` ranges are set with that in mind.
- Boundaries at 1% of height make wide-winged shapes (Lorentzian, long EMG tails) have long [start, end] spans. That is by design; raise `boundary_frac` for tighter spans.
- `realistic` is tuned to PRIMAS: noise ~0.2–0.6 mAU, heights ~1.5–450 mAU, FWHM ~1–4 s.
