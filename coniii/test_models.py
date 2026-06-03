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


def test_int_multipliers():
    """Integer-typed multipliers must not corrupt energies/sampling (issue #34).

    Previously an int multiplier vector left `calc_e` doing integer arithmetic,
    which gave wrong energies and badly wrong samples. Models and samplers now
    coerce multipliers to float, so int and float inputs must behave identically.
    """
    from .samplers import Metropolis
    from .utils import define_ising_helper_functions

    n = 3
    vals = [0, 0, 0, 1, 1, 1]                       # h = 0, J = 1 (ferromagnetic)
    assert Ising(np.array(vals, dtype=int)).multipliers.dtype == np.float64

    calc_e = define_ising_helper_functions()[0]
    s_int = Metropolis(n, np.array(vals, dtype=int),   calc_e=calc_e, rng=np.random.RandomState(0))
    s_flt = Metropolis(n, np.array(vals, dtype=float), calc_e=calc_e, rng=np.random.RandomState(0))
    assert s_int.theta.dtype == np.float64
    s_int.generate_sample(500, n_iters=20, systematic_iter=True)
    s_flt.generate_sample(500, n_iters=20, systematic_iter=True)
    # identical rng + identical (now float) parameters -> identical samples
    assert np.array_equal(s_int.sample, s_flt.sample)


if __name__ == '__main__':
    pass
