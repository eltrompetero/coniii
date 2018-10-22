from coniii.samplers import FastMCIsing
import numpy as np
import time

def test_MCIsing():
    # Check that everything compiles.
    n=10
    print("Running timing suite for Metropolis sampling functions for n=%d..."%n)

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
    test_MCIsing()
