# ====================================================================================== #
# ConIII module for maxent models.
# Authors: Edward Lee (edlee@alumni.princeton.edu) and Bryan Daniels
#          (bryan.daniels.1@asu.edu)
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
# ====================================================================================== #
from importlib import import_module
import multiprocess as mp
from .utils import *
from .samplers import Metropolis
from .samplers import Potts3 as mcPotts3


class Model():
    """Basic model class outline.
    """
   
    def setup_sampler(self,
                      sample_method='metropolis',
                      sample_size=1000,
                      sampler_kwargs={}):
        """
        Instantiate sampler class object. Uses self.rng as the random number generator.

        Parameters
        ----------
        sample_method : str, 'metropolis'
            'metropolis'
        sample_size : int, 1000
        sampler_kwargs : dict, {}
            Kwargs that can be passed into the initialization function for the sampler.
        """
        
        self.sampleSize = sample_size

        if sample_method=='metropolis' and (type(self) is Ising or type(self) is Triplet):
            self.sampleMethod = sample_method
            self.sampler = Metropolis( self.n, self.multipliers, self.calc_e,
                                       n_cpus=self.nCpus,
                                       rng=self.rng,
                                       **sampler_kwargs )
        elif sample_method=='metropolis' and type(self) is Potts3:
            self.sampleMethod = sample_method
            self.sampler = mcPotts3( self.n, self.multipliers, self.calc_e,
                                     n_cpus=self.nCpus,
                                     rng=self.rng,
                                     **sampler_kwargs )
        else:
           raise NotImplementedError("Unrecognized sampler %s."%sample_method)
        self.sample = None

    def set_rng(self, rng):
        """Replace random number generator.

        Parameters
        ----------
        rng : np.random.RandomState
        """
        self.rng = rng
        self.sampler.rng = rng

    def generate_samples(self, n_iters, burn_in,
                         multipliers=None,
                         sample_size=None,
                         sample_method=None,
                         generate_kwargs={}):
        """
        Wrapper around generate_samples() generate_samples_parallel() methods in samplers.

        Samples are saved to self.sample.

        Parameters
        ----------
        n_iters : int
        burn_in : int 
        multipliers : ndarray, None
        sample_size : int, None
        sample_method : str, None
        generate_kwargs : dict, {}
        """

        assert not (self.sampler is None), "Must call setup_sampler() first."
        
        if multipliers is None:
            multipliers = self.multipliers
        sample_method = sample_method or self.sampleMethod
        sample_size = sample_size or self.sampleSize
        
        # When sequential sampling should be used.
        if not self.nCpus is None and self.nCpus<=1:
            if sample_method=='metropolis':
                self.sampler.theta = multipliers.copy()
                # Burn in.
                self.sampler.generate_samples(sample_size,
                                              n_iters=burn_in)
                self.sampler.generate_samples(sample_size,
                                              n_iters=n_iters)
                self.sample = self.sampler.samples

            else:
               raise NotImplementedError("Unrecognized sampler.")
        # When parallel sampling using the multiprocess module.
        else:
            if sample_method=='metropolis':
                self.sampler.theta = multipliers.copy()
                self.sampler.generate_samples_parallel(sample_size,
                                                       n_iters=burn_in+n_iters)
                self.sample = self.sampler.samples

            else:
               raise NotImplementedError("Unrecognized sampler.")
#end Model


class Ising(Model):
    """Ising model parameterized by fields and couplings.
    """
    def __init__(self, multipliers, rng=None, n_cpus=None, verbose=False):
        """
        Parameters
        ----------
        multipliers : list of ndarray or ndarray
            Can be an integer (all parameters are set to zero), list of vectors [fields,
            couplings], a vector of fields and couplings concatenated together, or a
            matrix of parameters where the diagonal entries are the fields.
        """
        
        # intelligently read in multipliers by handling multiple use cases
        if type(multipliers) is int:
            assert multipliers>1, "System size must be greater than 1."
            self.n = multipliers
            multipliers = np.zeros(self.n+self.n*(self.n-1)//2)
        if len(multipliers)==2:
            # case where two vectors are given
            self.n = len(multipliers[0])
            assert self.n*(self.n-1)/2==len(multipliers[1]), "Must be n fields and (n choose 2) couplings."
            multipliers = np.concatenate(multipliers)
        elif type(multipliers) is np.ndarray and multipliers.ndim==2:
            # case where matrix is given
            multipliers = multipliers.copy()
            self.n = multipliers.shape[0]

            h = multipliers.diagonal().copy()
            multipliers[np.diag_indices(multipliers.shape[0])] = 0
            multipliers = np.concatenate([h, squareform(multipliers)]) 
        elif type(multipliers) is np.ndarray:
            # case where all parameters are given in single vector
            self.n = int((np.sqrt(1+8*multipliers.size)-1)//2)

            assert multipliers.size==(self.n+(self.n-1)*self.n/2), "Incompatible dimensions for multipliers."
        else:
            raise Exception("Unrecognized format for multipliers.")
        
        self.calc_e, _, _ = define_ising_helper_functions()
        try:
            ising = import_module('coniii.ising_eqn.ising_eqn_%d_sym'%self.n)
            self._calc_observables = ising.calc_observables
            self._calc_p = ising.p
        except ModuleNotFoundError:
            self._calc_observables = None
            self._calc_p = None
            self.calc_observables = None
            self.calc_p = None
        self.set_multipliers(multipliers)

        self.rng = rng or np.random.RandomState()  # this will get passed to sampler if it is set up
        self.nCpus = n_cpus
        self.verbose = verbose

    def set_multipliers(self, multipliers):
        """Set multipliers to a new value. Need to redefine some functions that rely on
        copy of self.multipliers.
        """

        self.multipliers = multipliers

        # if system is small enough, we can use enumeration to calculate observables from the multipliers
        if not self._calc_observables is None:
            self.calc_observables = lambda x=self.multipliers: self._calc_observables(x)
            self.calc_p = lambda x=self.multipliers: self._calc_p(x)
#end Ising

# alias for Ising
PairwiseMaxent = Ising


class Triplet(Model):
    """Third order maxent model constraining means, pairwise correlations, and triplet
    correlations.
    """
    def __init__(self, multipliers, rng=None, n_cpus=None, verbose=False):
        """
        Parameters
        ----------
        multipliers : list of ndarray or ndarray
            Can be a list of vectors [fields, couplings], a vector of fields and couplings
            concatenated together, or a matrix of parameters where the diagonal entries
            are the fields.
        """
        
        # parameters must be given separately
        self.n = len(multipliers[0])
        assert binom(self.n,2)==len(multipliers[1]), "Wrong number of couplings."
        assert binom(self.n,3)==len(multipliers[2]), "Wrong number of triplet interactions."
        multipliers = np.concatenate(multipliers)
        
        self.calc_e = define_triplet_helper_functions()[0]
        try:
            ising = import_module('coniii.ising_eqn.ising_eqn_%d_sym_triplet'%self.n)
            self._calc_observables = ising.calc_observables
            self._calc_p = ising.p
        except ModuleNotFoundError:
            self._calc_observables = None
            self._calc_p = None
            self.calc_observables = None
            self.calc_p = None
        self.set_multipliers(multipliers)

        self.rng = rng or np.random.RandomState()  # this will get passed to sampler if it is set up
        self.nCpus = n_cpus
        self.verbose = verbose

    def set_multipliers(self, multipliers):
        """Set multipliers to a new value. Need to redefine some functions that rely on
        copy of self.multipliers.
        """

        self.multipliers = multipliers

        # if system is small enough, we can use enumeration to calculate observables from the multipliers
        if not self._calc_observables is None:
            self.calc_observables = lambda x=self.multipliers: self._calc_observables(x)
            self.calc_p = lambda x=self.multipliers: self._calc_p(x)
#end Triplet


class Potts3(Model):
    """Three-state spin model constraining means and pairwise correlations.
    """
    def __init__(self, multipliers, rng=None, n_cpus=None, verbose=False):
        """
        Parameters
        ----------
        multipliers : list of ndarray
            Can be a list of vectors [fields, couplings].
        """
        
        if type(multipliers) is list:
            assert (len(multipliers[0])%3)==0
            assert len(multipliers)==2

            # parameters must be given separately
            self.n = int(len(multipliers[0])//3)
            assert binom(self.n,2)==len(multipliers[1]), "Wrong number of couplings."
            multipliers = np.concatenate(multipliers)
        else:
            assert type(multipliers) is np.ndarray, "Multipliers must be ndarray or list."
            n = (np.sqrt(25+8*multipliers.size)-5)/2
            assert n==int(n), "Incompatible number of parameters."
            self.n = int(n)
        
        self.calc_e = define_potts_helper_functions(3)[0]
        try:
            ising = import_module('coniii.ising_eqn.ising_eqn_%d_potts'%self.n)
            self._calc_observables = ising.calc_observables
            self._calc_p = ising.p
        except ModuleNotFoundError:
            self._calc_observables = None
            self._calc_p = None
            self.calc_observables = None
            self.calc_p = None
        self.set_multipliers(multipliers)

        self.rng = rng or np.random.RandomState()  # this will get passed to sampler if it is set up
        self.nCpus = n_cpus
        self.verbose = verbose

    def set_multipliers(self, multipliers):
        """Set multipliers to a new value. Need to redefine some functions that rely on
        copy of self.multipliers.
        """

        self.multipliers = multipliers

        # if system is small enough, we can use enumeration to calculate observables from the multipliers
        if not self._calc_observables is None:
            self.calc_observables = lambda x=self.multipliers: self._calc_observables(x)
            self.calc_p = lambda x=self.multipliers: self._calc_p(x)
#end Potts3
