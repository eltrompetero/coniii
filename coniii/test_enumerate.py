# ========================================================================================================= #
# Test suite for enumerate.py
# Author : Edward Lee, edlee@alumni.princeton.edu
#
# MIT License
# 
# Copyright (c) 2017 Edward D. Lee, Bryan C. Daniels
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================= #
import numpy as np
from .utils import pair_corr, bin_states
from .enumerate import fast_logsumexp
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

def test_fast_logsumexp():
    from scipy.special import logsumexp

    X = np.random.normal(size=10, scale=10, loc=1000)
    coeffs = np.random.choice([-1,1], size=X.size)

    assert np.array_equal(fast_logsumexp(X, coeffs), logsumexp(X, b=coeffs, return_sign=True))
