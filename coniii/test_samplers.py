# =============================================================================================== #
# Testing for samplers.py module.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
#
# MIT License
# 
# Copyright (c) 2020 Edward D. Lee, Bryan C. Daniels
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
import numpy as np
import multiprocess as mp
import pytest
import time

from .samplers import *
from . import samplers as samplers_module
from .utils import define_ising_helper_functions, define_potts_helper_functions

n = 5


def test_sample_ising():
    assert sample_ising(np.zeros(n+n*(n-1)//2), 100).shape==(100,n)

def test_Metropolis(run_timing=False):
    # Check that everything compiles and runs.
    theta = np.random.normal(size=15, scale=.1)
    calc_e, _, _ = define_ising_helper_functions()
    print("Running timing suite for Metropolis sampling functions for n=%d..."%n)

    sampler = Metropolis(n, theta, calc_e)
    print("Running sampler.generate_sample(n)")
    sampler.generate_sample(100)
    print("Done.")

    print("Running sampler.generate_sample(n, systematic_iter=True)")
    sampler.generate_sample(100, systematic_iter=True)
    print("Done.")

    # test control over rng
    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample(100, systematic_iter=True)
    X1 = sampler.sample.copy()

    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample(100, systematic_iter=True)
    X2 = sampler.sample.copy()

    assert np.array_equal(X1, X2), (X1, X2)
   
    # parallelization
    print("Running sampler.generate_sample_parallel(100, systematic_iter=True)")
    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=True)
    X1 = sampler.sample.copy()

    print("Running sampler.generate_sample_parallel(100, systematic_iter=True)")
    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=True)
    X2 = sampler.sample.copy()
    assert np.array_equal(X1, X2)

    # parallelization without systematic iter
    print("Running sampler.generate_sample_parallel(100, systematic_iter=False)")
    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=False)
    X1 = sampler.sample.copy()

    print("Running sampler.generate_sample_parallel(100, systematic_iter=False)")
    sampler = Metropolis(n, theta, calc_e, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=False)
    X2 = sampler.sample.copy()
    assert np.array_equal(X1, X2)

    if run_timing:
        # Some basic timing checks
        print("Timing sequential sampling")
        sampler = Metropolis(n, theta, calc_e)
        t0=time.perf_counter()
        sampler.generate_sample(100, n_iters=10000, systematic_iter=True)
        print(time.perf_counter()-t0)
        
        print("Timing parallel sampling")
        sampler = Metropolis(n, theta, calc_e)
        t0=time.perf_counter()
        sampler.generate_sample_parallel(100, n_iters=10000, systematic_iter=True)
        print(time.perf_counter()-t0)

def test_Potts3():
    np.random.seed(0)

    # Check that everything compiles and runs.
    n = 5
    theta = np.random.normal(size=n*3 + n*(n-1)//2, scale=.1)
    calc_e = define_potts_helper_functions(3)[0]
    print("Running timing suite for Potts3 sampling functions for n=%d..."%n)

    sampler = Potts3(n, theta, calc_e, n_cpus=1)
    print("Running sampler.generate_sample(n)")
    sampler.generate_sample(100)
    assert set(np.unique(sampler.sample)) <= {0, 1, 2}
    print("Done.")

    print("Running sampler.generate_sample(n, systematic_iter=True)")
    sampler.generate_sample(100, systematic_iter=True)
    assert set(np.unique(sampler.sample)) <= {0, 1, 2}
    print("Done.")

    # test control over rng
    sampler = Potts3(n, theta, calc_e, n_cpus=1, rng=np.random.RandomState(0))
    sampler.generate_sample(100, n_iters=10, systematic_iter=True)
    X1 = sampler.sample.copy()

    sampler = Potts3(n, theta, calc_e, n_cpus=1, rng=np.random.RandomState(0))
    sampler.generate_sample(100, n_iters=10, systematic_iter=True)
    X2 = sampler.sample.copy()
    assert np.array_equal(X1, X2), (X1[:10], X2[:10])
   
    # parallelization
    print("Running sampler.generate_sample_parallel(100, systematic_iter=True)")
    sampler = Potts3(n, theta, calc_e, n_cpus=2, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=True)
    X1 = sampler.sample.copy()
    assert set(np.unique(X1)) <= {0, 1, 2}

    print("Running sampler.generate_sample_parallel(100, systematic_iter=True)")
    sampler = Potts3(n, theta, calc_e, n_cpus=2, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=True)
    X2 = sampler.sample.copy()
    assert np.array_equal(X1, X2)

    # parallelization without systematic iter
    print("Running sampler.generate_sample_parallel(100, systematic_iter=False)")
    sampler = Potts3(n, theta, calc_e, n_cpus=2, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=False)
    X1 = sampler.sample.copy()

    print("Running sampler.generate_sample_parallel(100, systematic_iter=False)")
    sampler = Potts3(n, theta, calc_e, n_cpus=2, rng=np.random.RandomState(0))
    sampler.generate_sample_parallel(100, systematic_iter=False)
    X2 = sampler.sample.copy()
    assert np.array_equal(X1, X2)

def test_WolffIsing():
    """The Wolff single-cluster sampler should reproduce the exact distribution.

    Regression: the jitted helper iterate_neighbors built its output with
    np.zeros(dtype=int), which modern numba refuses to compile, so the sampler
    crashed on first use.
    """
    from .ising_eqn import ising_eqn_3_sym as ising
    from .utils import pair_corr

    rng = np.random.RandomState(0)
    nn = 3
    h = rng.normal(scale=.4, size=nn)
    J = rng.normal(scale=.4, size=nn*(nn-1)//2)             # mixed-sign couplings
    exact = ising.calc_observables(np.concatenate((h, J)))

    # NOTE: the WolffIsing constructor takes (couplings, fields), in that order.
    w = WolffIsing(J, h)
    w.rng = np.random.RandomState(0)
    X = w.generate_sample(30000, n_iters=30)
    assert X.shape == (30000, nn)
    assert set(np.unique(X).tolist()) <= {-1., 1.}
    # empirical means and pairwise correlations match the exact ones (the field
    # acceptance and arbitrary-sign bond rule are exercised here)
    emp = pair_corr(X, concat=True)
    assert np.abs(exact - emp).max() < 0.02, np.abs(exact - emp)

    # serial sampling is reproducible given a seeded rng
    wa = WolffIsing(J, h); wa.rng = np.random.RandomState(1)
    wb = WolffIsing(J, h); wb.rng = np.random.RandomState(1)
    assert np.array_equal(wa.generate_sample(200, n_iters=10),
                          wb.generate_sample(200, n_iters=10))

    # the parallel path runs and produces valid output
    Xp = WolffIsing(J, h).generate_sample_parallel(500, n_iters=10)
    assert Xp.shape == (500, nn)
    assert set(np.unique(Xp).tolist()) <= {-1., 1.}


def test_ParallelTempering():
    """Replica exchange should run end to end and the target (beta=1) replica
    should reproduce the exact distribution."""
    from .ising_eqn import ising_eqn_3_sym as ising
    from .utils import pair_corr

    rng = np.random.RandomState(0)
    nn = 3
    h = rng.normal(scale=.4, size=nn)
    J = rng.normal(scale=.4, size=nn*(nn-1)//2)
    hJ = np.concatenate((h, J))
    exact = ising.calc_observables(hJ)
    calc_e = define_ising_helper_functions()[0]

    pt = ParallelTempering(nn, hJ, calc_e, 3, (1., 3.), rng=np.random.RandomState(0))
    # temperature ladder: strictly increasing beta with the target fixed at beta == 1
    assert (np.diff(pt.beta) > 0).all(), pt.beta
    assert np.isclose(pt.beta[-1], 1.), pt.beta

    pt.generate_sample(2500)
    # samples are stored per replica; the target distribution is the highest-beta one
    assert len(pt.sample) == 3
    Xtarget = pt.sample[-1]
    assert Xtarget.shape == (2500, nn)
    assert set(np.unique(Xtarget).tolist()) <= {-1, 1}
    # mean error over observables is a stable statistic (less noisy than the max)
    emp = pair_corr(Xtarget, concat=True)
    assert np.abs(exact - emp).mean() < 0.03, np.abs(exact - emp)


def _energy_history_checks(sampler, theta, calc_e, n_iters):
    """Serial nopython sampling: energies, history and starting state."""
    history = sampler.generate_sample(50, n_iters=n_iters, saveHistory=True)
    assert np.allclose(sampler.E, calc_e(sampler.sample, theta))
    assert history.size == 50*n_iters + 1
    assert np.allclose(history[n_iters-1:-1:n_iters], sampler.E)
    assert np.array_equal(sampler._samples, sampler.sample[-1:])

    s0 = sampler.random_sample(1)
    sampler.generate_sample(3, n_iters=0, burn_in=0, initial_sample=s0)
    assert np.array_equal(sampler.sample, np.tile(s0, (3, 1)))
    with pytest.raises(AssertionError):
        sampler.generate_sample(3, initial_sample=sampler.random_sample(2))


def _parallel_checks(make_sampler, theta, calc_e):
    """Parallel nopython sampling: one chain per CPU, types as in the Python path,
    continuation and seeded reproducibility."""
    n_cpus = mp.cpu_count()
    sampler = make_sampler(1)
    sampler.generate_sample_parallel(101, n_iters=10)
    assert sampler.sample.shape == (101, sampler.n)
    assert sampler.sample.dtype == np.int8
    assert sampler._samples.shape == (n_cpus, sampler.n)
    assert type(sampler.E) is tuple and len(sampler.E) == n_cpus
    assert np.allclose(sampler.E, calc_e(sampler._samples, theta))

    # continuing from self._samples is the same as passing the final states back in
    a = make_sampler(1)
    a.generate_sample_parallel(101, n_iters=10)
    a.generate_sample_parallel(101, n_iters=10)
    b = make_sampler(1)
    b.generate_sample_parallel(101, n_iters=10)
    b.generate_sample_parallel(101, n_iters=10, initial_sample=b._samples.copy())
    assert np.array_equal(a.sample, b.sample)

    c = make_sampler(2)
    c.generate_sample_parallel(101, n_iters=10, systematic_iter=True)
    d = make_sampler(2)
    d.generate_sample_parallel(101, n_iters=10, systematic_iter=True)
    assert np.array_equal(c.sample, d.sample)


def test_Metropolis_njit(monkeypatch):
    """Without the C++ extension, pairwise models use the nopython samplers with the
    same arguments and attributes as the pure-Python path."""
    monkeypatch.setattr(samplers_module, "IMPORTED_SAMPLERS_EXT", False)
    theta = np.random.RandomState(0).normal(size=n+n*(n-1)//2, scale=.3)
    calc_e = define_ising_helper_functions()[0]

    # calc_e is filled in when not given
    sampler = Metropolis(n, theta, rng=np.random.RandomState(0), iprint=False)
    assert sampler.generate_sample.__name__ == "generate_sample_njit"
    assert sampler.generate_sample_parallel.__name__ == "generate_sample_parallel_njit"
    assert sampler.generate_cond_sample.__name__ == "generate_cond_sample_py"
    assert sampler.calc_e is not None

    _energy_history_checks(sampler, theta, calc_e, 10)
    assert set(np.unique(sampler.sample).tolist()) <= {-1, 1}

    X = [None, None]
    for i in range(2):
        sampler = Metropolis(n, theta, rng=np.random.RandomState(0), iprint=False)
        sampler.generate_sample(100, n_iters=10)
        X[i] = sampler.sample
    assert np.array_equal(*X)

    _parallel_checks(lambda seed: Metropolis(n, theta, rng=np.random.RandomState(seed),
                                             iprint=False),
                     theta, calc_e)

    # conditional sampling stays in Python and works without calc_e being passed
    sampler = Metropolis(n, theta, rng=np.random.RandomState(0), iprint=False)
    sampler.generate_cond_sample(20, [(0, 1)], burn_in=10, n_iters=10, parallel=False)
    assert sampler.sample.shape == (20, n)

    # sample_ising passes the documented keyword arguments through
    X = sample_ising(theta, 100, seed=0, parallel=False,
                     generate_sample_kw={'saveHistory': True,
                                         'initial_sample': np.ones((1, n), dtype=int)})
    assert X.shape == (100, n)
    assert sample_ising(theta, 100, seed=0).shape == (100, n)


def test_Potts3_njit(monkeypatch):
    """Potts3 counterpart of test_Metropolis_njit."""
    monkeypatch.setattr(samplers_module, "IMPORTED_SAMPLERS_EXT", False)
    theta = np.random.RandomState(0).normal(size=3*n+n*(n-1)//2, scale=.3)
    calc_e = define_potts_helper_functions(3)[0]

    sampler = Potts3(n, theta, rng=np.random.RandomState(0))
    assert sampler.generate_sample.__name__ == "generate_sample_njit"
    assert sampler.generate_sample_parallel.__name__ == "generate_sample_parallel_njit"
    assert sampler.calc_e is not None

    _energy_history_checks(sampler, theta, calc_e, 10)
    assert set(np.unique(sampler.sample).tolist()) <= {0, 1, 2}

    # theta as (fields, couplings) gives the same samples as the concatenated vector
    a = Potts3(n, theta, rng=np.random.RandomState(0))
    a.generate_sample(100, n_iters=10)
    b = Potts3(n, (theta[:3*n], theta[3*n:]), rng=np.random.RandomState(0))
    b.generate_sample(100, n_iters=10)
    assert np.array_equal(a.sample, b.sample)

    _parallel_checks(lambda seed: Potts3(n, theta, rng=np.random.RandomState(seed)),
                     theta, calc_e)


def test_models_njit(monkeypatch):
    """The solvers sample through Model.generate_sample; check it on the nopython path."""
    from .models import Ising, Potts3 as Potts3Model
    monkeypatch.setattr(samplers_module, "IMPORTED_SAMPLERS_EXT", False)

    for model in (Ising(np.random.RandomState(0).normal(size=n+n*(n-1)//2, scale=.3),
                        rng=np.random.RandomState(0)),
                  Potts3Model(np.random.RandomState(0).normal(size=3*n+n*(n-1)//2, scale=.3),
                              rng=np.random.RandomState(0))):
        model.setup_sampler(sample_size=100)
        assert model.sampler.generate_sample.__name__ == "generate_sample_njit"
        for parallel in (False, True):
            model.generate_sample(10, 10, parallel=parallel)
            assert model.sample.shape == (100, n)


#if __name__=='__main__':
def compare_samplers():
    import time


    # check for similarity in results between different samplers
    np.random.seed(0)
    theta = np.random.normal(size=n+n*(n-1)//2, scale=.1)
    calc_e,_,_ = define_ising_helper_functions()
    nSamples = 1_000
    
    t0 = time.perf_counter()
    sampler1 = ParallelTempering(n, theta, calc_e, 4, (1,3), replica_burnin=n*100, rep_ex_burnin=n*10)
    sampler1.generate_sample(nSamples, save_exchange_trajectory=True)
    print("Sampler 1 took %1.2f s."%(time.perf_counter()-t0))
    
    t0 = time.perf_counter()
    sampler2 = Metropolis(n, theta, calc_e)
    sampler2.generate_sample_parallel(nSamples)
    print("Sampler 2 took %1.2f s."%(time.perf_counter()-t0))
    
    from .ising_eqn.ising_eqn_5_sym import calc_observables
    print(sampler1.sample[-1].mean(0), sampler2.sample.mean(0), calc_observables(theta)[:n]) 
    print(sampler1.sample[-1].std(axis=0)/np.sqrt(nSamples), sampler2.sample.std(axis=0)/np.sqrt(nSamples)) 

    return sampler1, sampler2
