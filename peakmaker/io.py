"""Dataset files: chrom_XXXXX.npz (uv, fs) + chrom_XXXXX.json (labels, params) + manifest.json."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_chromatogram(out_dir, index, result):
    stem = Path(out_dir) / f"chrom_{index:05d}"
    np.savez(f"{stem}.npz", uv=result["uv"], fs=result["fs"])
    meta = {k: v for k, v in result.items() if k != "uv"}
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")


def load_chromatogram(json_path):
    """Return (uv, fs, meta) for a chrom_XXXXX.json path."""
    json_path = Path(json_path)
    d = np.load(json_path.with_suffix(".npz"))
    return d["uv"], float(d["fs"]), json.loads(json_path.read_text(encoding="utf-8"))


def save_manifest(out_dir, preset, cfg, master_seed, n):
    tpl = cfg["real_templates"]
    tpl_hash = hashlib.sha256(Path(tpl).read_bytes()).hexdigest() if os.path.exists(tpl) else None
    manifest = {"preset": preset, "master_seed": master_seed, "n": n,
                "real_templates_sha256": tpl_hash, "config": cfg}
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
