"""Parametric peak shape library.

Every shape is returned as a *unit template*: sampled on the normalized axis U,
apex at u=0, apex height 1, FWHM 1. Warping/placement (warp.py) treats all
templates -- parametric and real -- the same way.
"""
import numpy as np
from scipy.stats import exponnorm

from .rand import uniform

U_MAX = 40.0
U = np.linspace(-U_MAX, U_MAX, 8001)
_X = np.linspace(-100.0, 100.0, 40001)  # raw grid for building shapes (sigma units)

COMPOSITES = ("shoulder", "doublet")


def normalize(x, y):
    """Resample a raw single-peak curve onto U: apex at u=0, height 1, FWHM 1.

    Raises IndexError if the curve does not drop below half height on both sides.
    """
    y = y / y.max()
    i = int(np.argmax(y))
    l = np.nonzero(y[:i] < 0.5)[0][-1]
    r = i + np.nonzero(y[i:] < 0.5)[0][0]
    xl = x[l] + (0.5 - y[l]) / (y[l + 1] - y[l]) * (x[l + 1] - x[l])
    xr = x[r - 1] + (y[r - 1] - 0.5) / (y[r - 1] - y[r]) * (x[r] - x[r - 1])
    return np.interp(U, (x - x[i]) / (xr - xl), y, left=0.0, right=0.0)


def gaussian(rng, p):
    return normalize(_X, np.exp(-0.5 * _X**2)), {}


def emg_tail(rng, p):
    k = uniform(rng, p["tau_ratio"])  # tau / sigma
    return normalize(_X, exponnorm.pdf(_X, k)), {"tau_ratio": k}


def emg_front(rng, p):
    tpl, sp = emg_tail(rng, p)
    return tpl[::-1].copy(), sp  # U is symmetric, so reversing mirrors the shape


def bigaussian(rng, p):
    ratio = uniform(rng, p["ratio"])  # wide side sigma / narrow side sigma
    if rng.random() < 0.5:
        ratio = 1.0 / ratio
    sigma = np.where(_X < 0, 1.0, ratio)
    return normalize(_X, np.exp(-0.5 * (_X / sigma) ** 2)), {"right_left_ratio": ratio}


def lorentzian(rng, p):
    eta = uniform(rng, p["eta"])  # pseudo-Voigt: 1 = pure Lorentzian, 0 = pure Gaussian
    y = eta / (1 + 4 * _X**2) + (1 - eta) * np.exp(-4 * np.log(2) * _X**2)
    return normalize(_X, y), {"eta": eta}


def flat_top(rng, p):
    power = uniform(rng, p["power"])  # super-Gaussian; 1 = Gaussian, higher = flatter top
    return normalize(_X, np.exp(-0.5 * np.abs(_X) ** (2 * power))), {"power": power}


SHAPES = {
    "gaussian": gaussian,
    "emg_tail": emg_tail,
    "emg_front": emg_front,
    "bigaussian": bigaussian,
    "lorentzian": lorentzian,
    "flat_top": flat_top,
    "broad_hump": bigaussian,  # same shape, own (wide) fwhm_s range in the preset
    "narrow": gaussian,        # same shape, own (narrow) fwhm_s range in the preset
}
