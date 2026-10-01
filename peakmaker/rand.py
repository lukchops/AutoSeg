"""Small sampling helpers shared by the generator modules."""
import numpy as np


def uniform(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def loguniform(rng, lo_hi):
    return float(np.exp(rng.uniform(np.log(lo_hi[0]), np.log(lo_hi[1]))))


def randint(rng, lo_hi):
    """Integer in [lo, hi], both inclusive."""
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))
