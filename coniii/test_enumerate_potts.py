# ===================================================================================== #
# Test suite for enumerate_potts.py
# Author : Edward Lee, edlee@alumni.princeton.edu
#
# MIT License
# 
# Copyright (c) 2019 Edward D. Lee, Bryan C. Daniels
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
# ===================================================================================== #
import numpy as np
import mpmath as mp
from .utils import pair_corr, bin_states
from .enumerate_potts import *
import os
import importlib
np.random.seed(0)


def single_pass(n,k):
    # write equations file for testing
    writer = SpecificFieldGenericCouplings(n, k)
    writer.write('ising_eqn/_test_enumerate_potts%d.py'%n)

    ising = importlib.import_module('.ising_eqn._test_enumerate_potts%d'%n, package='coniii')
    hJ = np.random.normal(size=n*k+n*(n-1)//2, scale=.2)
    
    p = ising.p(hJ)
    assert np.isclose(p.sum(), 1)
    print("Test passed: Normalized probability.")
    assert ((ising.calc_observables(hJ)<=1)&(ising.calc_observables(hJ)>=0)).all()
    print("Test passed: correlations bounded in [0,1].")

    allStates = np.vstack(list(xpotts_states(n,k))).astype(np.int8)
    sisj = ising.calc_observables(hJ)
    si = sisj[:k*n]
    sisj = sisj[n*k:]
    for i in range(n):
        for k_ in range(k):
            assert np.isclose(si[n*k_+i], (allStates[:,i]==k_).dot(p))
    for ijix,(i,j) in enumerate(combinations(range(n),2)):
        assert np.isclose(sisj[ijix], (allStates[:,i]==allStates[:,j]).dot(p))
    print("Test passed: correlations calculated from probability distribution agree with direct calculation.")

def test_basic():
    try:
        n = 3
        k = 3
        single_pass(n,k)
    finally:    
        # cleanup
        os.remove('ising_eqn/_test_enumerate_potts%d.py'%n)
    
    try:
        n = 4
        k = 3
        single_pass(n,k)
    finally:    
        # cleanup
        os.remove('ising_eqn/_test_enumerate_potts%d.py'%n)
