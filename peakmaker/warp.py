"""Random deformation of unit templates and placement onto the sample grid."""
import numpy as np

from .shapes import U


def shape_noise(rng, tpl, amp):
    """Smooth multiplicative deformation (a few low-frequency sines), apex renormalized to 1."""
    if amp <= 0:
        return tpl
    k = 3
    freq = rng.uniform(0.05, 0.5, k)  # cycles per FWHM
    phase = rng.uniform(0, 2 * np.pi, k)
    a = rng.normal(0, 1, k) / np.sqrt(k)
    pert = (a[:, None] * np.sin(2 * np.pi * freq[:, None] * U + phase[:, None])).sum(0)
    out = tpl * (1 + amp * pert)
    return out / out.max()


def place(tpl, n, fs, apex_s, fwhm_s, asym):
    """Evaluate a unit template on n samples at fs Hz.

    asym > 1 stretches the left half by asym and shrinks the right half by asym (and vice versa).
    """
    u = (np.arange(n) / fs - apex_s) / fwhm_s
    u = np.where(u < 0, u / asym, u * asym)
    return np.interp(u, U, tpl, left=0.0, right=0.0)
