"""Load YAML presets."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("fs", "length_min", "peak_count", "boundary_frac", "real_templates",
            "peaks", "families", "background", "artifacts")


def load_preset(name):
    """`name` is a preset name in presets/ (e.g. "realistic") or a path to a .yaml file."""
    path = Path(name) if name.endswith((".yaml", ".yml")) else ROOT / "presets" / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    cfg["real_templates"] = str(ROOT / cfg["real_templates"])
    return cfg
