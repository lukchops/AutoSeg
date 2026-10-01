"""Curated real peak templates (written by pick_templates.py)."""
import os

import numpy as np

from .shapes import U


def load_real_templates(path):
    """Return a list of (name, unit template on U). Empty if the file does not exist yet."""
    if not os.path.exists(path):
        print(f"warning: no real templates at {path} -- 'real' family disabled (run pick_templates.py)")
        return []
    d = np.load(path)
    return [(str(name), np.interp(U, d["u"], tpl, left=0.0, right=0.0))
            for name, tpl in zip(d["names"], d["templates"])]
