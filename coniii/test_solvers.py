# ====================================================================================== #
# Testing for solvers.py module. See usage_guide.ipynb for comprehensive examples of
# testing algorithms on test data.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .solvers import *
from .utils import (bin_states, pair_corr, logsumexp,
                    define_ising_helper_functions,
                    define_pseudo_ising_helper_functions,
                    unique_rows)
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
get_multipliers_r,calc_observables_r = define_pseudo_ising_helper_functions(n)


def test_init():
    """Check that all derived Solver classes can be initialized."""
    from .utils import pair_corr, define_ising_helper_functions, define_pseudo_ising_helper_functions
    
    # Define function specifically needed for creating Enumerate class.
    def calc_observables_multipliers(J):
        """
        Calculate observables from probability distribution given Langrangian multipliers. For
        the Ising model, these are the means of each spin and the pairwise correlations.
        """
        E = calc_e(allstates, J)
        return pair_corr( allstates, np.exp(-E-logsumexp(-E)), concat=True )
    
    # try initializing the solvers
    solver = Enumerate(sample)
    solver = SparseEnumerate(sample, parameter_ix=np.array([0,1,2]))
    solver = MPF(sample)
    solver = Pseudo(sample)
    solver = ClusterExpansion(sample)
    solver = MCH(sample, sample_size=1000)
    solver = RegularizedMeanField(sample)

    solver = Enumerate(n)
    solver = SparseEnumerate(n, parameter_ix=np.array([0,1,2]))
    solver = MPF(n)
    solver = Pseudo(n)
    solver = ClusterExpansion(n)
    solver = MCH(n, sample_size=1000)
    solver = RegularizedMeanField(n)

def test_Enumerate():
    from .utils import pair_corr

    # Enumerate should be able to find exact solution when passed the exact correlations
    solver = Enumerate(sample)
    soln = solver.solve(initial_guess=hJ/2, constraints=sisjTrue)
    assert np.isclose(hJ, soln).all()

def test_SparseEnumerate():
    from .utils import pair_corr

    # SparseEnumerate should be able to find exact solution when passed the exact correlations
    solver = SparseEnumerate(sample, parameter_ix=np.array([0,1,2,3,4,5]))
    soln = solver.solve(initial_guess=hJ/2, constraints=sisjTrue)
    assert np.isclose(hJ, soln).all()

def test_MPF():
    """Check MPF."""

    solver = MPF(sample)
    
    # compare log objective function with non-log version
    # Convert from {0,1} to {+/-1} basis.
    X = sample
    X = (X+1)//2
     
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
    assert np.isclose( solver.solve(),
                       solver.solve(uselog=False),
                       atol=1e-3 ).all()

def test_Pseudo():
    solver = Pseudo(sample)
    solver.solve(initial_guess=np.zeros(6))
    # atol=2e-2: pseudo-likelihood doesn't fit empirical correlations
    # exactly, so a tight tolerance from sample sisj is not the right check
    assert np.isclose(ising.calc_observables(solver.multipliers), sisj, atol=2e-2).all()

    solver.solve(force_general=True, initial_guess=np.zeros(6))
    assert np.isclose(ising.calc_observables(solver.multipliers), sisj, atol=2e-2).all()


def test_MCH():
    """Smoke test: MCH solver runs and produces multipliers of the expected shape."""
    solver = MCH(sample, sample_size=1000, iprint=False)
    soln = solver.solve(initial_guess=np.zeros(6),
                        maxiter=2,
                        n_iters=5,
                        burn_in=5,
                        iprint=False)
    multipliers = solver.multipliers
    assert multipliers.shape == (6,)
    assert np.isfinite(multipliers).all()


def test_ClusterExpansion():
    """Smoke test: ClusterExpansion solver runs end-to-end."""
    solver = ClusterExpansion(sample, sample_size=1000, iprint=False)
    soln = solver.solve(threshold=0.1, iprint=False)
    multipliers = solver.multipliers
    assert multipliers.shape == (6,)
    assert np.isfinite(multipliers).all()


def test_RegularizedMeanField():
    """Smoke test: RegularizedMeanField solver runs over a coarse grid."""
    solver = RegularizedMeanField(sample, sample_size=1000, iprint=False)
    soln = solver.solve(n_grid_points=5)
    multipliers = solver.multipliers
    assert multipliers.shape == (6,)
    assert np.isfinite(multipliers).all()


def test_Pseudo_hessian():
    """Pseudo now supplies an analytic Hessian and solves with Newton-CG by default
    (issue #21). On the smooth 'general' objective the Hessian-based optimum must match
    a gradient-only (BFGS) optimum, which confirms the Hessian is correct.
    """
    nparams = n + n*(n-1)//2

    s_hess = Pseudo(sample, iprint=False)
    s_hess.solve(force_general=True, initial_guess=np.zeros(nparams))             # Newton-CG + hess
    s_grad = Pseudo(sample, iprint=False)
    s_grad.solve(force_general=True, initial_guess=np.zeros(nparams),
                 solver_kwargs={'method': 'BFGS'})                                # gradient only
    assert np.allclose(s_hess.multipliers, s_grad.multipliers, atol=1e-3), \
        np.abs(s_hess.multipliers - s_grad.multipliers).max()

    # the default (per-spin Ising) path also converges to a finite solution
    s_ising = Pseudo(sample, iprint=False)
    s_ising.solve(initial_guess=np.zeros(nparams))
    assert np.isfinite(s_ising.multipliers).all()


def test_MCH_reproducible():
    """MCH with a fixed rng must be deterministic (issue #28)."""
    def run():
        solver = MCH(sample, sample_size=1000, rng=np.random.RandomState(0), iprint=False)
        solver.solve(initial_guess=np.zeros(n*(n-1)//2 + n),
                     maxiter=3, n_iters=20, burn_in=50, iprint=False)
        return solver.multipliers.copy()
    a = run()
    b = run()
    assert np.array_equal(a, b), np.abs(a - b).max()


def test_RegularizedMeanField_degenerate():
    """RMF must not crash on (near-)degenerate data (issue #16).

    Near-perfectly-correlated spins drive the mean-field J toward singular/nan,
    which used to crash scipy's bracketing (or squareform). The solver should now
    return a finite solution instead.
    """
    rng = np.random.RandomState(3)
    X = rng.choice([-1, 1], size=(2000, 3))
    X[:, 1] = np.where(rng.rand(2000) < 0.97, X[:, 0], X[:, 1])  # corr(s0,s1) ~ 0.94
    solver = RegularizedMeanField(X, sample_size=500, iprint=False)
    out = solver.solve()
    assert out.shape == (n*(n-1)//2 + n,)
    assert np.isfinite(out).all()


def test_pickling():
    pass

if __name__=='__main__':
    pass
