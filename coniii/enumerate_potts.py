# ====================================================================================== #
# Enumerate module for writing equations for Potts models. There are many different kinds
# of ways of parameterizing Potts models especially in the face of limited data. Here are
# a few particular examples that may be useful.
# 
# Provided as part of the ConIII package.
# 
# Author: Eddie Lee, edlee@alumni.princeton.edu
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
import sys
import numpy as np
import re
from datetime import datetime
from .enumerate import fast_logsumexp
from .utils import xpotts_states
from itertools import combinations


# ========= #
# Functions #
# ========= #
def split_string(s, n):
    """Insert \n character every n.
    """
    
    i = n
    while i<len(s):
        s = s[:i]+'\n'+s[i:]
        i += n+2

def insert_newlines(s, n):
    """Insert \n character every n in list.
    """
    
    i = n
    while i<len(s):
        s.insert(i, '\n')
        i += n+2


# ======= #
# Classes #
# ======= #
class PythonFileWriterBase():
    def __init__(self, n, k):
        """
        Parameters
        ----------
        n : int
            System size.
        k : int
            Number of discrete states.
        """
        
        assert n>2
        assert k>=2

        self.n = n
        self.k = k

    def energy_terms_generator(self):
        """Generator for iterating through all possible states and yield the energy expression as well as
        the configuration of spins.
        """
        return

    def write(self, fname):
        """Write equations to file.
        """

        with open(fname, 'w') as f:
            self._write_header(f)
            self._write_correlations(f)
            self._write_probabilities(f)

    def _write_header(self, f):
        """
        Parameters
        ----------
        f : file
        """
        
        n, k = self.n, self.k

        # write intro
        f.write('# This file contains equations for the Potts model with %d spins and %d states.\n'%(n,k))
        f.write('# Provided as part of ConIII package.\n')
        f.write('# Written on %s.\n'%str(datetime.now()))
        f.write('#\n#\n')
        
        # read in license file and copy it
        try:
            licenseText = ''.join(['# '+el for el in open('LICENSE.txt','r').readlines()])
            f.write(licenseText)
        except FileNotFoundError:
            print("LICENSE.txt not found.")
        f.write('#\n')
            
        # import statements
        f.write('from numpy import zeros, exp\n')
        f.write('from ..enumerate import fast_logsumexp\n\n')

    def _write_correlations(self, f):
        f.write('def calc_observables(params):\n')
        f.write('\tassert params.size==%d\n'%(self.n*self.k+self.n*(self.n-1)//2))
        f.write('\th = params[:%d]\n'%(self.n*self.k))
        f.write('\tJ = params[%d:]\n'%(self.n*self.k))
        f.write('\tobservables = zeros(%d)\n'%(self.n*self.k+self.n*(self.n-1)//2))

        # write energy terms
        allStates = []
        f.write('\tenergies = [')
        for term in self.energy_terms_generator():
            f.write('\t\t%s,\n'%term[0])
            allStates.append(term[1])
        f.write('\t\t]\n')

        # write partition function
        f.write('\tlogZ = fast_logsumexp(energies)[0]\n')
        
        # write means
        counter = 0
        for k in range(self.k):
            for i in range(self.n):
                siCoeffs = []
                for s in allStates:
                    if s[i]==str(k):
                        siCoeffs.append('1')
                    else:
                        siCoeffs.append('0')
                insert_newlines(siCoeffs, 55)
                siCoeffs = ','.join(siCoeffs).replace('\n,','\n')
                f.write('\tobservables[%d] = exp(fast_logsumexp(energies, [%s])[0] - logZ)\n'%(counter,
                                                                                               siCoeffs))
                counter += 1
        # write all pairwise correlations
        for i,j in combinations(range(self.n),2):
            sisjCoeffs = []
            for s in allStates:
                if s[i]==s[j]:
                    sisjCoeffs.append('1')
                else:
                    sisjCoeffs.append('0')
            insert_newlines(sisjCoeffs, 55)
            sisjCoeffs = ','.join(sisjCoeffs).replace('\n,','\n')
            f.write('\tobservables[%d] = exp(fast_logsumexp(energies, [%s])[0] - logZ)\n'%(counter,
                                                                                           sisjCoeffs))
            counter += 1
        f.write('\treturn observables')
        f.write('\n\n')

    def _write_probabilities(self, f):
        f.write('def p(params):\n')
        f.write('\tassert params.size==%d\n'%(self.n*self.k+self.n*(self.n-1)//2))
        f.write('\th = params[:%d]\n'%(self.n*self.k))
        f.write('\tJ = params[%d:]\n'%(self.n*self.k))
        f.write('\tp = zeros(%d)\n'%(self.k**self.n))

        # write energy terms
        f.write('\tenergies = [')
        for term in self.energy_terms_generator():
            f.write('\t\t%s,\n'%term[0])
        f.write('\t\t]\n')

        # write partition function
        f.write('\tlogZ = fast_logsumexp(energies)[0]\n')
        
        # write all probabilities
        for i,term in enumerate(self.energy_terms_generator()):
            f.write('\tp[%d] = exp(energies[%d] - logZ)\n'%(i,i))
        f.write('\treturn p\n')
#end PythonFileWriterBase


class SpecificFieldGenericCouplings(PythonFileWriterBase):
    """This version specifies a field for every distinct Potts state, but considers
    correlation averaged over all possible states as long as they agree. 
    
    When writing the equations, the fields are assumed to first increase by spin index
    then by state index (fields for the n spins for k=0 come first, then for k=1, etc.).
    The couplings come after in the conventional order of ij where j increases up to i
    before i is incremented.
    """
    def energy_terms_generator(self):
        """Generator for iterating through all possible states and yield the energy
        expression as well as the configuration of spins. The energy expression is
        returned as a string and assumes that the field come first followed by the
        couplings.
        """
        
        n, k = self.n, self.k

        def gen():
            for s in xpotts_states(n, k):
                term = ''
                for i in range(n):
                    term += 'h[%d]+'%(int(s[i])*n+i)

                for counter,(i,j) in enumerate(combinations(range(n),2)):
                    if s[i]==s[j]:
                        term += 'J[%d]+'%counter

                yield term[:-1], s
        return gen()
#end SpecificFieldGenericCouplings


# terminal interface
if __name__=='__main__':
    """An example to write an N=9 system with K=3 distinct states. Must specify that it is
    a module to make relative imports work (see below):
    >>> python -m coniii.enumerate_potts 9 3 coniii/ising_eqn/ising_eqn_9_potts.py
    """

    n = int(sys.argv[1])
    k = int(sys.argv[2])
    fname = sys.argv[3]
    assert n>2 and k>=2

    writer = SpecificFieldGenericCouplings(n, k)
    writer.write(fname)
