"""Graph and coarse-graining helpers.

Internal module split out of the original coniii.utils. Public names
are re-exported by :mod:`coniii.utils`.
"""
import numpy as np
from numba import njit
from scipy.spatial.distance import squareform
from warnings import warn

@njit
def adj(s, n_random_neighbors=0):
    """Return one-flip neighbors and a set of random neighbors. This is written to be used
    with the solvers.MPF class. Use adj_sym() if symmetric spins in {-1,1} are needed.
    
    NOTE: For random neighbors, there is no check to make sure neighbors don't repeat but
    this shouldn't be a problem as long as state space is large enough.

    Parameters
    ----------
    s : ndarray
        State whose neighbors are found. One-dimensional vector of spins.
    n_random_neighbors : int,0
        If >0, return this many random neighbors. Neighbors are just random states, but
        they are called "neighbors" because of the terminology in MPF. They can provide
        coupling from s to states that are very different, increasing the equilibration
        rate.

    Returns
    -------
    neighbors : ndarray
        Each row is a neighbor. s.size + n_random_neighbors are returned.
    """

    neighbors = np.zeros((s.size+n_random_neighbors,s.size))
    for i in range(s.size):
        s[i] = 1-s[i]
        neighbors[i] = s.copy()
        s[i] = 1-s[i]
    if n_random_neighbors:
        for i in range(n_random_neighbors):
            match = True
            while match:
                newneighbor = (np.random.rand(s.size)<.5)*1.
                # Make sure neighbor is not the same as the given state.
                if (newneighbor!=s).any():
                    match=False
            neighbors[i+s.size] = newneighbor
    return neighbors

@njit
def adj_sym(s, n_random_neighbors=False):
    """Symmetric version of adj() where spins are in {-1,1}.
    """

    neighbors = np.zeros((s.size+n_random_neighbors,s.size))
    for i in range(s.size):
        s[i] = -1*s[i]
        neighbors[i] = s.copy()
        s[i] = -1*s[i]
    if n_random_neighbors:
        for i in range(n_random_neighbors):
            match=True
            while match:
                newneighbor=(np.random.rand(s.size)<.5)*2.-1
                # Make sure neighbor is not the same as the given state.
                if (newneighbor!=s).any():
                    match=False
            neighbors[i+s.size]=newneighbor
    return neighbors


def coarse_grain_with_func(X, n_times, sim_func, coarse_func):
    """Iteratively coarse-grain X by combining pairs with the highest similarity. Both the
    function to measure similarity and to implement the coarse-graining must be supplied.

    Parameters
    ----------
    X : ndarray
        Each col is a variable and each row is an observation (n_samples, n_system).
    n_times : int
        Number of times to coarse grain.
    sim_func : function
        Takes an array like X and returns a vector of ncol*(ncol-1)//2 pairwise
        similarities.
    coarse_func : function
        Takes a two col array and returns a single vector.

    Returns
    -------
    ndarray
        Coarse-grained version of X.
    list of lists of ints 
        Each list specifies which columns of X have been coarse-grained into each col of
        the coarse X.
    """

    assert np.log2(X.shape[1])>n_times

    coarseX = X.copy()
    originalIx = [[i] for i in range(X.shape[1])]

    # Combine sets of spins with the largest pairwise correlations
    for coarseix in range(n_times):
        n = coarseX.shape[1]
        cij = squareform(sim_func(coarseX))
        assert cij.shape==(n,n)
        ix = list(range(coarseX.shape[1]))
        
        newClusters = []
        for i in range(n//2):
            # find maximally correlated pair of spins
            mxix = np.argmax(cij.ravel())
            mxix = (mxix//(n-2*i), mxix%(n-2*i))  # row and col
            if mxix[0]>mxix[1]:
                mxix = (mxix[1],mxix[0])
            
            newClusters.append((ix[mxix[0]], ix[mxix[1]]))
            # remove corresponding rows and cols of combined pair
            cij = np.delete(np.delete(cij, mxix[0], axis=0), mxix[0], axis=1)
            cij = np.delete(np.delete(cij, mxix[1]-1, axis=0), mxix[1]-1, axis=1)
            ix.pop(mxix[0])
            ix.pop(mxix[1]-1)
        if n%2:
            # if X contains an odd number of cols
            newClusters.append((ix[0],))
        
        # coarse-grain data
        X_ = np.zeros((coarseX.shape[0],int(np.ceil(n/2))), dtype=X.dtype)
        originalIx_ = []
        for i,ix in enumerate(newClusters):
            X_[:,i] = coarse_func(coarseX[:,ix])
            originalIx_.append([])
            for ix_ in ix:
                originalIx_[-1] += originalIx[ix_]
        originalIx = originalIx_
        coarseX = X_
    binsix = originalIx
    
    return coarseX, binsix

