# =============================================================================================== #
# Testing for samplers.py module.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
#
# MIT License
# 
# Copyright (c) 2017 Edward D. Lee
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
from .samplers import FastMCIsing,Metropolis
from .utils import define_ising_helper_functions
import numpy as np
import time


def test_Metropolis():
    # Check that everything compiles and runs.
    n=10
    theta=np.random.normal(size=55)
    calc_e, _, _ = define_ising_helper_functions()
    print("Running timing suite for Metropolis sampling functions for n=%d..."%n)

    sampler=Metropolis(n, theta, calc_e)
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

    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=True)
    print("Done.")

    print("Running sampler.generate_samples_parallel(n, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=False)
    print("Done.")
    print()

def test_FastMCIsing(run_timing=False):
    # Check that everything compiles and runs.
    n=10
    print("Running timing suite for Ising sampling functions for n=%d..."%n)

    theta=np.random.normal(size=55)
    sampler=FastMCIsing(n, theta)
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

    print("Running sampler.generate_samples(n, saveHistory=True, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=True)
    print("Done.")

    print("Running sampler.generate_samples_parallel(n, systematic_iter=True)")
    sampler.generate_samples_parallel(n, systematic_iter=False)
    print("Done.")
    print()
    
    if run_timing:
        # Some basic timing checks
        print("Timing jit loop")
        t0=time.time()
        sampler.generate_samples(10, n_iters=10000, systematic_iter=True)
        print(time.time()-t0)

        print("Timing parallel jit loop")
        t0=time.time()
        sampler.generate_samples_parallel(10, n_iters=10000, systematic_iter=True)
        print(time.time()-t0)

        print("Timing pure Python loop")
        sampler=FastMCIsing(n, theta, use_numba=False)
        t0=time.time()
        sampler.generate_samples(10, n_iters=10000, systematic_iter=True)
        print(time.time()-t0)

        print("Timing parallel pure Python loop")
        t0=time.time()
        sampler.generate_samples_parallel(10, n_iters=10000, systematic_iter=True)
        print(time.time()-t0)

if __name__=='__main__':
    test_FastMCIsing(True)
