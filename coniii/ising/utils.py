# ===================================================================================== #
# Module with useful functions for Ising models.
# Distributed as part of ConIII.
# Author : Edward Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
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
import numpy as np
from scipy.spatial.distance import squareform
import itertools
import importlib 
from ..utils import *


class Ising():
    """A nice front end for the pairwise maxent (Ising) model in the {-1,1} basis.
    """
    def __init__(self, n, h=None, J=None):
        """
        Parameters
        ----------
        n : int
            System size.
        h : list-like, None
            Fields.
        J : list-like, None
            Couplings.
        """
        
        # check args
        assert n>1
        if h is None:
            h = np.zeros(n)
        elif not hasattr(h, '__len__'):
            h = np.zeros(n)+h
        else:
            assert len(h)==n, "Number of fields should be equal to n."
        if J is None:
            J = np.zeros(n*(n-1)//2)
        elif not hasattr(J, '__len__'):
            J = np.zeros(n*(n-1)//2)+J
        else:
            assert len(J)==(n*(n-1)//2), "Number of couplings should be equal to n choose 2."
        assert h.ndim==1 and J.ndim==1, "Both h and J must be provided as vectors."
        
        self.n = n
        self.hJ = np.concatenate((h,J))
        self.Jmat = squareform(J)
        self.hJ01 = convert_params(h, J, '01', concat=True)
        self.ising_eqns = importlib.import_module('coniii.ising_eqn.ising_eqn_%d_sym'%n)
        self.calc_e,_,_ = define_ising_helper_functions()

    def correlations(self, basis='1'):
        """
        Parmeters
        ---------
        basis : str, '1'

        Returns
        -------
        ndarray
            Means and pairwise correlations.
        """

        sisj = self.ising_eqns.calc_observables(self.hJ)
        if basis=='1':
            return sisj
        return convert_corr(sisj[:self.n], sisj[self.n:], convert_to='01', concat=True)

    def fields(self, basis='1'):
        """
        Parameters
        ----------
        basis : str, '1'
            '0' or '1'

        Returns
        -------
        ndarray
        """

        if basis=='1':
            return self.hJ[:self.n]
        return self.hJ01[:self.n]

    def couplings(self, basis='1'):
        """
        Parameters
        ----------
        basis : str, '1'
            '0' or '1'

        Returns
        -------
        ndarray
        """

        if basis=='1':
            return self.hJ[self.n:]
        return self.hJ01[self.n:]

    def find_basin(self, s):
        """Return energy basins for given state using single spin flips.

        Parameters
        ----------
        s : ndarray

        Returns
        -------
        ndarray
        """
        
        assert s.size==self.n
        atMin = False
        thisState = s.astype(np.int8)

        while not atMin:        
            dE = self.neighbor_dE(thisState)
            if np.any( dE<0 ):
                ix = dE.argmin()
                thisState[ix] *= -1
            else:
                atMin = True
        return thisState
    
    def neighbor_dE(self, state):
        """dE to get to single flip neighbors."""

        dE = np.zeros(self.n)
        for i in range(self.n):
            dE[i] = 2*state[i]*self.hJ[i] +2*state[i]*(state*self.Jmat[i]).sum()
        return dE
    
    @staticmethod
    def resort_couplings(J,sortIx):
        """Reorder given couplings into a desired order.
        
        Params:
        -------
        J (ndarray)
            vector of length n*(n-1)/2
        sortIx (ndarray)
        """
        return
#end Ising
