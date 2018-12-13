# =============================================================================================== #
# Testing for samplers.py module.
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
from .samplers import *
from .utils import define_ising_helper_functions
import numpy as np
import time

n=5


def test_Metropolis(run_timing=False):
    # Check that everything compiles and runs.
    n=5
    theta=np.random.normal(size=15, scale=.1)
    calc_e, _, _ = define_ising_helper_functions()
    print("Running timing suite for Metropolis sampling functions for n=%d..."%n)

    sampler=Metropolis(n, theta, calc_e, n_cpus=1)
    print("Running sampler.generate_samples(n)")
    sampler.generate_samples(n)
    print("Done.")

    print("Running sampler.generate_samples(n, systematic_iter=True)")
    sampler.generate_samples(n, systematic_iter=True)
    print("Done.")

    print("Running sampler.generate_samples(n, saveHistory=True)")
    sampler.generate_samples(n, saveHistory=True)
    print("Done.")

    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
    sampler.generate_samples(n, saveHistory=True, systematic_iter=True)
    print("Done.")

     # test control over rng
    sampler.rng = np.random.RandomState(0)
    initialSample = np.random.choice([-1.,1], size=(1,n))
    sampler.generate_samples(n, systematic_iter=True, initial_sample=initialSample)
    X1 = sampler.samples.copy()

    sampler.rng = np.random.RandomState(0)
    sampler.generate_samples(n, systematic_iter=True, initial_sample=initialSample)
    X2 = sampler.samples.copy()

    assert np.array_equal(X1, X2), (X1, X2)
   
    # parallelization
    sampler = Metropolis(n, theta, calc_e)
    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=True)
    print("Done.")

    print("Running sampler.generate_samples_parallel(n, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=False)
    print("Done.")

    if run_timing:
        # Some basic timing checks
        print("Timing sequential sampling")
        sampler = Metropolis(n, theta, calc_e, n_cpus=1)
        t0=time.perf_counter()
        sampler.generate_samples(10, n_iters=10000, systematic_iter=True)
        print(time.perf_counter()-t0)
        
        print("Timing parallel sampling")
        sampler = Metropolis(n, theta, calc_e)
        t0=time.perf_counter()
        sampler.generate_samples_parallel(10, n_iters=10000, systematic_iter=True)
        print(time.perf_counter()-t0)

#def test_FastMCIsing(run_timing=False):
#    # Check that everything compiles and runs.
#    n=5
#    print("Running timing suite for Ising sampling functions for n=%d..."%n)
#    theta=np.random.normal(size=15, scale=.2)
#    
#    sampler=FastMCIsing(n, theta)
#    print("Running sampler.generate_samples(n)")
#    sampler.generate_samples(n)
#    print("Done.")
#
#    print("Running sampler.generate_samples(n, systematic_iter=True)")
#    sampler.generate_samples(n, systematic_iter=True)
#    print("Done.")
#
#    print("Running sampler.generate_samples(n, saveHistory=True)")
#    sampler.generate_samples(n, saveHistory=True)
#    print("Done.")
#
#    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
#    sampler.generate_samples(n, saveHistory=True, systematic_iter=True)
#    print("Done.")
#
#    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
#    sampler.generate_samples_parallel(n, systematic_iter=True)
#    print("Done.")
#
#    print("Running sampler.generate_samples_parallel(n, systematic_iter=True)")
#    sampler.generate_samples_parallel(n, systematic_iter=False)
#    print("Done.")
#    print()
#
#    # test that setting rng reproduces same sample
#    sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0))
#    sampler.rng = np.random.RandomState(0)
#    sampler.generate_samples(5, n_iters=2, systematic_iter=False)
#    X1 = sampler.samples.copy()
#
#    sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0))
#    sampler.rng = np.random.RandomState(0)
#    sampler.generate_samples(5, n_iters=2, systematic_iter=False)
#    X2 = sampler.samples.copy()
#    assert np.array_equal(X1, X2)
#
#    sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0))
#    sampler.rng = np.random.RandomState(0)
#    sampler.generate_samples_parallel(5, n_iters=2, systematic_iter=True)
#    X1 = sampler.samples.copy()
#    
#    sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0))
#    sampler.generate_samples_parallel(5, n_iters=2, systematic_iter=True)
#    X2 = sampler.samples.copy()
#    assert np.array_equal(X1, X2)
#
#    if run_timing:
#        # Some basic timing checks
#        print("Timing sequential sampling")
#        sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0), n_cpus=1)
#        t0=time.perf_counter()
#        sampler.generate_samples(10, n_iters=10000, systematic_iter=True)
#        print(time.perf_counter()-t0)
#
#        print("Timing parallel sampling")
#        sampler = FastMCIsing(n, theta, rng=np.random.RandomState(0))
#        t0=time.perf_counter()
#        sampler.generate_samples_parallel(10, n_iters=10000, systematic_iter=True)
#        print(time.perf_counter()-t0)

def test_ParallelTempering():
    # basic functionality
    theta = np.random.normal(size=n+n*(n-1)//2, scale=.1)
    calc_e,_,_ = define_ising_helper_functions()
    sampler = ParallelTempering(n, theta, calc_e, 4, (1,3))
    sampler.generate_samples(10)


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
    sampler1.generate_samples(nSamples, save_exchange_trajectory=True)
    print("Sampler 1 took %1.2f s."%(time.perf_counter()-t0))
    
    t0 = time.perf_counter()
    sampler2 = Metropolis(n, theta, calc_e)
    sampler2.generate_samples_parallel(nSamples)
    print("Sampler 2 took %1.2f s."%(time.perf_counter()-t0))
    
    from .ising_eqn.ising_eqn_5_sym import calc_observables
    print(sampler1.samples[-1].mean(0), sampler2.samples.mean(0), calc_observables(theta)[:n]) 
    print(sampler1.samples[-1].std(axis=0)/np.sqrt(nSamples), sampler2.samples.std(axis=0)/np.sqrt(nSamples)) 

    return sampler1, sampler2
