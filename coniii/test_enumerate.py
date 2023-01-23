# ===================================================================================== #
# Test suite for enumerate.py
# Author : Edward Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
import mpmath as mp
from .utils import pair_corr, bin_states
from .enumerate import fast_logsumexp, mp_fast_logsumexp
np.random.seed(0)


def test_basic():
    hJ = np.random.normal(size=6,scale=.2)
    
    # make sure probability distribution is normalized, p and correlations agree for both symmetrized and
    # unsymmetrized bases
    # n=3
    from .ising_eqn import ising_eqn_3_sym as ising
    p = ising.p(hJ)
    assert np.isclose(p.sum(), 1)
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=-1)).all()
    assert np.isclose(ising.calc_observables(hJ),
                      pair_corr(bin_states(3,True), weights=ising.p(hJ), concat=True)).all()

    from .ising_eqn import ising_eqn_3 as ising
    p = ising.p(hJ)
    assert np.isclose(p.sum(), 1)
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=0)).all()
    assert np.isclose(ising.calc_observables(hJ),
                      pair_corr(bin_states(3), weights=ising.p(hJ), concat=True)).all()

    
    # n=4
    hJ = np.random.normal(size=10, scale=.2)

    from .ising_eqn import ising_eqn_4_sym as ising
    p = ising.p(hJ)
    assert np.isclose(p.sum(), 1)
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=-1)).all()
    assert np.isclose(ising.calc_observables(hJ),
                      pair_corr(bin_states(4,True), weights=ising.p(hJ), concat=True)).all()

    from .ising_eqn import ising_eqn_4 as ising
    p = ising.p(hJ)
    assert np.isclose(p.sum(), 1)
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=0)).all()
    assert np.isclose(ising.calc_observables(hJ),
                      pair_corr(bin_states(4), weights=ising.p(hJ), concat=True)).all()

    # n=4, high precision
    hJ = np.array(list(map(mp.mpf, np.random.normal(size=10, scale=.2))))

    from .ising_eqn import ising_eqn_4_sym_hp as ising
    p = ising.p(hJ)
    assert np.isclose(float(p.sum()), 1)
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=-1)).all()
    assert np.isclose(ising.calc_observables(hJ).astype(float),
                      pair_corr(bin_states(4,sym=True), weights=ising.p(hJ).astype(float), concat=True)).all()

def test_fast_logsumexp():
    from scipy.special import logsumexp

    X = np.random.normal(size=10, scale=10, loc=1000)
    coeffs = np.random.choice([-1,1], size=X.size)
    
    npval = logsumexp(X, b=coeffs, return_sign=True)
    assert np.array_equal(fast_logsumexp(X, coeffs), npval)

    X = np.array(list(map(mp.mpf, X)))
    assert abs(float(mp_fast_logsumexp(X, coeffs)[0])-npval[0])<1e-16
