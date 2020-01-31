# For profiling different variations on Metropolis sampler.
from coniii.samplers import Metropolis
from coniii.utils import define_ising_helper_functions
import numpy as np

calc_e = define_ising_helper_functions()[0]
n = 10
multipliers = np.random.normal(size=n+n*(n-1)//2, scale=.1)

sboost = Metropolis(n, multipliers, calc_e, boost=True)
spy = Metropolis(n, multipliers, calc_e)

# copy following lines into ipython
#%timeit sboost.generate_samples(100)
#%timeit spy.generate_samples(100)
#%timeit sboost.generate_samples_parallel(100)
#%timeit spy.generate_samples_parallel(100)
