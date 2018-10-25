# Class for storing data for solving with Ising methods.
# Author : Edward Lee, edlee@alumni.princeton.edu
# 2015-04-30
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

import numpy as np
from misc_fcns import *
import workspace.utils as ws
from scipy.spatial.distance import squareform
import entropy.entropy as entropy
from . import fast
import itertools

class Data():
    """
    Class for keeping the data and any notes.
    2015-05-05
    """
    def __init__(self,binary,mask=None,notes=''):
        self.binary = binary # binary data {0,1}
        self.sym = 2*self.binary-1 # symmetrized formulation of binary data
        self.mask = mask
        self.notes = notes
        self.N = self.binary.shape[1]
        #self.oData = # original form of data

        self.si,self.sisj = entropy.calc_sisj(self.binary)
        return

    def get_binary(self,sym=False):
        if sym:
            return self.binary*2-1
        else:
            return self.binary
    def get_correl():
        """Return all correlations as list."""
        return [self.si,self.sisj]


class IsingModel():
    """
    Given details in {0,1} formulation, makes retrieval of basic quantities easy.
    2015-07-12
    """

    def __init__(self,N,J,data,correls=None):
        import importlib 
        assert 11>N>0
        assert len(J)==(N*(N-1)/2+N)
         
        tosolve = importlib.import_module('tosolve01.tosolve%d'%N)
        self.N = N
        self.J = J # all parameters. bias field first
        self.data = data
        self.binStates = entropy.bin_states(self.N)
        self.E = np.array([fast.calc_e( self.J, s[None,:] ) for s in entropy.xbin_states(self.N)]).ravel()
        
        if N<11 and correls is None:
            self.correls = tosolve.get_stats(self.J)

        self.h1,self.J1 = self.convert_params( self.hi('0'),self.Jij('0'),convertTo='11' )
    
    def si(self,form='1'):
        if form=='1':
            return self.si(form='0')*2-1.
        return self.correls[:self.N]

    def sisj(self,form='1'):
        if form=='1':
            return entropy.convert_sisj( self.sisj(form='0'), self.si(form='0'), '11' )
        return self.correls[self.N:]

    def hi(self,form='1'):
        if form=='1':
            return self.h1
        return self.J[:self.N]

    def Jij(self,form='1'):
        if form=='1':
            return self.J1
        return self.J[self.N:]

    def identify_basins(self):
        """
        Find all the minima in the energy landscape
        2015-08-19
        """
        dMat = np.zeros((2**self.N,2**self.N))
        for (i,j) in itertools.combinations(list(range(2**self.N)),2):
            dMat[i,j] = np.sum(np.abs(self.binStates[i].astype(float)-self.binStates[j]))
        dMat += dMat.T
        singleIx = dMat==1

        ix = np.zeros((2**self.N))
        for i,s in enumerate(entropy.bin_states(self.N,sym=True)):
            dE = self.E[dMat[i]==1] - self.E[i]
            if np.all(dE>0):
                ix[i] = 1
                
        basinsIx = ix==1
        return self.binStates[basinsIx]

    def find_basin(self,s):
        """
        Return energy basins for given state.
        """
        atMin = False
        neighborEnergies = np.zeros((self.N))
        currentState = s.copy()
        currentEnergy = self.E[ np.sum(currentState[None,:]==self.binStates,1)==self.N ]

        while not atMin:        
            neighborEnergies = self.neighbor_energies(currentState)
            dE = neighborEnergies - currentEnergy
            if np.any( dE<0 ):
                ix = dE.argmin()
                currentState[ix] = 1 - currentState[ix]
                currentEnergy = neighborEnergies[ix]
            else:
                atMin = True
        return (np.sum(currentState==self.binStates,1)==self.N).nonzero()[0]
    
    def neighbor_energies(self,currentState):
        neighborEnergies = np.zeros((self.N))
        for i in range(self.N):
            neighborState = currentState.copy()
            neighborState[i] = 1 - neighborState[i]
            neighborEnergies[i] = self.E[ self.state_ix(neighborState).nonzero()[0] ]
        return neighborEnergies
    
    def state_ix(self,s):
        """Return the index of the state.
        2015-07-12"""
        if s.ndim==1:
            return np.sum( s[None,:]==self.binStates,1 )==self.N
        else:
            return np.sum( s==self.binStates,1 )==self.N

    @staticmethod   
    def convert_params(h,J,convertTo='01'):
        """
            Convert parameters from 0,1 formulation to +/-1 and vice versa.
        2014-05-12
        """
        from entropy.entropy import squareform

        if len(J.shape)!=2:
            Jmat = squareform(J)
        else:
            Jmat = J
            J = squareform(J)
        
        if convertTo=='11':
            # Convert from 0,1 to -/+1
            Jp = J/4.
            hp = h/2 + np.sum(Jmat,1)/4.
        elif convertTo=='01':
            # Convert from -/+1 to 0,1
            hp = 2.*(h - np.sum(Jmat,1))
            Jp = J*4.

        return hp,Jp
    
    @staticmethod
    def resort_couplings(J,sortIx):
        """
        Reorder given couplings into a desired order.
        2015-07-12
        
        Params:
        -------
        J (ndarray)
            vector of length N*(N-1)/2
        sortIx (ndarray)
        """
        return
    
    def cij(self):
        """2015-07-12"""
        sisj = self.sisj()
        si = self.si()
        cij = []
        k = 0
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                cij.append( sisj[k] - si[i]*si[j] )
                k += 1
        return np.array(cij)
