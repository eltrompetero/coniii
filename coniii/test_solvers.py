# =============================================================================================== #
# Testing for solvers.py module. See usage_guide.ipynb for comprehensive examples of testing
# algorithms on test data.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
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
# =============================================================================================== #
from .solvers import *
from .ising_eqn import ising_eqn_3_sym as ising
import numpy as np
calc_observables_multipliers = ising.calc_observables

# Generate example data set to use in tests
n = 3  # system size
np.random.seed(0)  # standardize random seed
h = np.random.normal(scale=.1, size=n)           # random couplings
J = np.random.normal(scale=.1, size=n*(n-1)//2)  # random fields
hJ = np.concatenate((h, J))
p = ising.p(hJ)  # probability distribution of all states p(s)
sisjTrue = calc_observables_multipliers(hJ)  # exact means and pairwise correlations

allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n),
                                    size=10000,
                                    replace=True,
                                    p=p)]  # random sample from p(s)
sisj = pair_corr(sample, concat=True)  # means and pairwise correlations

# Define common functions.
calc_e, calc_observables, mchApproximation = define_ising_helper_functions()
get_multipliers_r,calc_observables_r = define_pseudo_ising_helpers(n)


def test_init():
    """Check that all derived Solver classes can be initialized."""
    from .utils import pair_corr, define_ising_helper_functions, define_pseudo_ising_helpers
    
    # Define function specifically needed for creating Enumerate class.
    def calc_observables_multipliers(J):
        """
        Calculate observables from probability distribution given Langrangian multipliers. For
        the Ising model, these are the means of each spin and the pairwise correlations.
        """
        E = calc_e(allstates, J)
        return pair_corr( allstates, np.exp(-E-logsumexp(-E)), concat=True )
    
    # try initializing the solvers
    solver = Enumerate(n,
                       calc_observables_multipliers=calc_observables_multipliers,
                       calc_observables=calc_observables)
    solver = MPF(n, 
                 calc_observables=calc_observables,
                 adj=adj)
    solver = Pseudo(n,
                    calc_observables=calc_observables,
                    calc_observables_r=calc_observables_r,
                    get_multipliers_r=get_multipliers_r)
    solver = ClusterExpansion(n, calc_observables=calc_observables)
    solver = MCH(n,
                 calc_observables=calc_observables,
                 sample_size=100,
                 sample_method='metropolis',
                 mch_approximation=mchApproximation,
                 n_cpus=1)
    solver = RegularizedMeanField(n, calc_observables=calc_observables)

def test_Enumerate():
    from .utils import pair_corr

    # Enumerate should be able to find exact solution when passed the exact correlations
    solver = Enumerate(n,
                       calc_observables_multipliers=calc_observables_multipliers,
                       calc_observables=calc_observables)
    soln = solver.solve(sisjTrue, initial_guess=hJ/2)
    assert np.isclose(hJ, soln).all()

def test_MPF():
    """Check MPF."""
    from .utils import adj

    solver = MPF(n, 
                 calc_observables=calc_observables,
                 adj=adj)
    
    # compare log objective function with non-log version
    # Convert from {0,1} to {+/-1} basis.
    X = sample
    X = (X+1)/2
     
    # Get list of unique data states and how frequently they appear.
    Xuniq = X[unique_rows(X)]
    ix = unique_rows(X, return_inverse=True)
    Xcount = np.bincount(ix)
    adjacentStates = solver.list_adjacent_states(Xuniq, True)

    # Interface to objective function.
    def f(params):
        return solver.logK( Xuniq, Xcount, adjacentStates, params )
    def g(params):
        return np.log( solver.K( Xuniq, Xcount, adjacentStates, params ) )
    
    # check that they evaluate to the same values once the log transform is accounted for
    assert np.isclose(f(hJ), g(hJ)), (f(hJ), g(hJ))
 
    # Check that found solutions agree closely
    assert np.isclose( solver.solve(sample, solver_kwargs={'disp':False}),
                       solver.solve(sample, solver_kwargs={'disp':False}, uselog=False),
                       atol=1e-3 ).all()

def test_Pseudo():
    solver = Pseudo(n,
                    calc_observables=calc_observables,
                    calc_observables_r=calc_observables_r,
                    get_multipliers_r=get_multipliers_r)
    estMultipliers1 = solver.solve(sample, np.zeros(6), general_case=True)
    estMultipliers2 = solver.solve(sample, np.zeros(6))
    
    # Check that both ways of solving the problem agree
    assert np.isclose(estMultipliers1, estMultipliers2, atol=1e-3).all()

def test_pickling():
    pass

if __name__=='__main__':
    test_MPF()
