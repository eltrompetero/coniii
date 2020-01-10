# =============================================================================================== #
# Module for simulations of Ising model on a lattice. Distributed as part of ConIII package.
# Author: Eddie Lee, edl56@cornell.edu
# =============================================================================================== #
import numpy as np
import multiprocess as mp
from numba import njit,jit


class Ising2D():
    """Simulation of the ferromagnetic Ising model on a 2D periodic lattice with quenched disorder
    in the local fields.
    """

    def __init__(self, dim, J, h=0, rng=None):
        """
        Parameters
        ----------
        dim : tuple
            Pair describing the length of the system along the x and y dimensions.
        J : float
        h : ndarray or float,0
            Field at every lattice point.
        rng : np.random.RandomState,None
        """
        
        assert len(dim)==2, "Must specify only x and y dimensions."
        self.dim = dim
        self.lattice = ((np.random.rand(*dim)<.5)*2.-1).astype(np.int8)
        self.J = J
        if type(h) is float or type(h) is type(int):
            self.h = np.zeros(dim)+h
        else:
            self.h = h or np.zeros(dim)
        self.rng = rng or np.random.RandomState()

    def iterate(self, n_iters, systematic=True):
        """
        Parameters
        ----------
        n_iters : int
        systematic : bool,True
            If True, iterate through each spin on the lattice in sequence.
        """

        flip_metropolis=self.flip_metropolis

        if not systematic:
            @njit
            def single_iteration(lattice, dim=self.dim, h=self.h, J=self.J):
                for i in range(n_iters):
                    i, j=np.random.randint(dim[0]), np.random.randint(dim[1])
                    flip_metropolis(i, j, h[i,j], J, lattice)
        else:
            # Fast iteration using jit.
            @njit
            def single_iteration(lattice, dim=self.dim, h=self.h, J=self.J, size=self.dim[0]*self.dim[1]):
                # Randomly order the lattice points for flipping or else lattice will be very important.
                ix = np.random.permutation( np.arange(dim[0]*dim[1]) )[:size]
                for ix_ in ix:
                    i, j = ix_//dim[1], ix_%dim[1]
                    flip_metropolis(i, j, h[i,j], J, lattice)

        lattice=self.lattice
        counter = 0
        while counter < n_iters:
            single_iteration(lattice)
            counter += self.dim[0]*self.dim[1]
        self.lattice = lattice

    @staticmethod
    @njit
    def flip_metropolis(i, j, h, J, lattice):
        """Flip a single lattice spin using Metropolis sampling.
        
        Parameters
        ----------
        i : int
        j : int
        """

        dE=0
        dim=lattice.shape

        # If same value as neighbor, will incur energy cost for flipping but if anti-aligned energy will
        # decrease
        if lattice[(i-1)%dim[0],j]==lattice[i,j]:
            dE+=2*J
        else:
            dE-=2*J
        if lattice[(i+1)%dim[0],j]==lattice[i,j]:
            dE+=2*J
        else:
            dE-=2*J
        if lattice[i,(j+1)%dim[1]]==lattice[i,j]:
            dE+=2*J
        else:
            dE-=2*J
        if lattice[i,(j-1)%dim[1]]==lattice[i,j]:
            dE+=2*J
        else:
            dE-=2*J
        
        # Local field.
        dE-=2*lattice[i,j]*h
        
        if dE<=0:
            lattice[i,j]*=-1
        elif np.random.rand()<np.exp(-dE): 
            lattice[i,j]*=-1
        return lattice 

def coarse_grain(lattice, factor):
    """Block spin renormalization with majority rule.
    
    Parameters
    ----------
    lattice : ndarray
        +/-1
    factor : int
    
    Returns
    -------
    renormalized_lattice : ndarray
    """

    reLattice=np.zeros((lattice.shape[0]//factor, lattice.shape[1]//factor), dtype=np.int16)
    reL=len(reLattice)

    for i in range(factor):
        for j in range(factor):
            reLattice+=lattice[i::factor,j::factor][:reL,:reL]
    reLattice=np.sign(reLattice)
    # when there is a tie, randomly choose a direction
    reLattice[reLattice==0]=np.random.choice([-1,1], size=(reLattice==0).sum())
    return reLattice
