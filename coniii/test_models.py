# ====================================================================================== #
# Tests for models.py construction.
# Released with ConIII package.
# ====================================================================================== #
import numpy as np

from .models import Ising, Triplet


def test_Ising_from_matrix():
    """Ising from a parameter matrix exercises scipy squareform (which models.py
    must import explicitly; it was previously leaked by `from .utils import *`)."""
    n = 5
    mat = np.zeros((n, n))
    solver = Ising(mat)
    assert solver.n == n
    assert solver.multipliers.shape == (n + n*(n-1)//2,)


def test_Triplet_construct():
    """Triplet construction exercises scipy binom (also previously leaked via the
    utils wildcard, then dropped from utils.__all__ in v4)."""
    n = 5
    model = Triplet([np.zeros(n),
                     np.zeros(n*(n-1)//2),
                     np.zeros(n*(n-1)*(n-2)//6)])
    assert model.n == n


if __name__ == '__main__':
    pass
