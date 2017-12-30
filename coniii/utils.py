from __future__ import division
import numpy as np
from numba import jit
from scipy.misc import logsumexp
from itertools import combinations
from scipy.spatial.distance import squareform



@jit(nopython=True,nogil=True,cache=True)
def sub_to_ind(n,i,j):
    """Convert pair of coordinates of a symmetric square array into consecutive index of flattened
    upper triangle. This is slimmed down so it won't throw errors like if i>n or j>n or if they're
    negative. Only checking for if the returned index is negative which could be problematic with
    wrapped indices.
    
    Parameters
    ----------
    n : int
        Dimension of square array
    i,j : int
        coordinates
    """
    if i<j:
        k = 0
        for l in xrange(1,i+2):
            k += n-l
        assert k>=0
        return k-n+j
    elif i>j:
        k = 0
        for l in xrange(1,j+2):
            k += n-l
        assert k>=0
        return k-n+i
    else:
        raise Exception("Indices cannot be the same.")

@jit(nopython=True,cache=True)
def ind_to_sub(n,ix):
    """Convert index from flattened upper triangular matrix to pair subindex.

    Parameters
    ----------
    n : int
        Dimension size of square array.
    ix : int
        Index to convert.

    Returns
    -------
    subix : tuple
        (i,j)
    """
    k = 0
    for i in xrange(n-1):
        for j in xrange(i+1,n):
            if k==ix:
                return (i,j)
            k += 1
 
def unique_rows(mat,return_inverse=False):
    """
    Return unique rows indices of a numeric numpy array.

    Params:
    -------
    mat (ndarray)
    **kwargs
    return_inverse (bool)
        If True, return inverse that returns back indices of unique array that would return the
        original array 

    Returns:
    --------
    u (ndarray)
        Unique elements of matrix.
    idx (ndarray)
        row indices of given mat that will give unique array
    """
    b = np.ascontiguousarray(mat).view(np.dtype((np.void, mat.dtype.itemsize * mat.shape[1])))
    if not return_inverse:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx = np.unique(b, return_inverse=True)
    
    return idx


def calc_overlap(sample,ignore_zeros=False):
    """
    <si_a si_b> between all pairs of replicas a and b

    Params:
    -------
    sample
    ignore_zeros (bool=False)
        Instead of normalizing by the number of spins, normalize by the minimum number of nonzero spins.
    """
    overlap = sample.dot(sample.T)
    if ignore_zeros:
        countZeros = np.zeros((len(sample),len(sample),2))
        countZeros[:,:,0] = (sample==0).sum(1)[:,None]
        countZeros[:,:,1] = (sample==0).sum(1)[None,:]
        return overlap / (sample.shape[1]-countZeros.max(2))
    return overlap / sample.shape[1]

def pair_corr(data,
              weights=None,
              concat=False,
              exclude_empty=False,
              subtract_mean=False):
    """
    Calculate averages and pairwise correlations of spins.

    Parameters
    ----------
    data : ndarray
        Dimensions (n_samples,n_dim).
    weights : np.ndarray,None : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one
    concat : bool,False
        return concatenated means if true
    exclude_empty : bool,False
        when using with {-1,1}, can leave entries with 0 and those will not be counted for any pair
        weights option doesn't do anything here
    subtract_mean : bool,False
        If True, return pairwise correlations with product of individual means subtracted.

    Returns
    -------
    (si,sisj) or np.concatenate((si,sisj))
    """
    S,N = data.shape
    sisj = np.zeros(N*(N-1)//2)
    
    if weights is None:
        weights = np.ones((data.shape[0]))/data.shape[0]

    if exclude_empty:
        assert np.array_equal( np.unique(data),np.array([-1,0,1]) ) or \
               np.array_equal( np.unique(data),np.array([-1,0]) ) or \
               np.array_equal( np.unique(data),np.array([0,1]) ) or \
               np.array_equal( np.unique(data),np.array([-1,1]) ) or \
               np.array_equal( np.unique(data),np.array([1]) ) or \
               np.array_equal( np.unique(data),np.array([-1]) ), "Only handles -1,1 data sets."
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = ( np.nansum(data[:,i]*data[:,j]) / 
                            (np.nansum(np.logical_and(data[:,i]!=0,data[:,j]!=0)) + np.nextafter(0,1)) )
                k+=1
        si = np.array([ np.nansum(col[col!=0]) / np.nansum(col!=0) for col in data.T ])
    else:
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = np.nansum(data[:,i]*data[:,j]*weights)
                k+=1
        si = np.nansum(data*weights[:,None],0)

    if subtract_mean:
        sisj = np.array([sisj[i]-si[ix[0]]*si[ix[1]] for i,ix in enumerate(combinations(range(N),2))])

    if concat:
        return np.concatenate((si,sisj))
    else:
        return si, sisj

def bin_states(n,sym=False):
    """
    Generate all possible binary spin states. 
    
    Parameters
    ----------
    n : int
        number of spins
    sym : bool
        if true, return {-1,1} basis

    Returns
    -------
    v : ndarray
    """
    if n<0:
        raise Exception("n cannot be <0")
    if n>30:
        raise Exception("n is too large to enumerate all states.")
    
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype('uint8')

    if sym is False:
        return v
    else:
        return v*-2.+1

def xbin_states(n,sym=False):
    """Generator for producing binary states.

    Parameters
    ----------
    n : int
        number of spins
    sym : bool
        if true, return {-1,1} basis
    """
    assert n>0, "n cannot be <0"
    
    def v():
        for i in xrange(2**n):
            if sym is False:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')
            else:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')*2.-1

    return v()

def convert_params(h,J,convertTo='01',concat=False):
    """
    Convert Ising model fields and couplings from {0,1} basis to {-1,1} and vice versa.

    Params:
    -------
    h (ndarray)
    J (ndarray)
    convertTo (str)
        '01' or '11'
    concat (bool=False)
        If True, return a vector concatenating fields and couplings.
    
    Returns:
    --------
    h (ndarray)
    J (ndarray)
    """
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

    if concat:
        return np.concatenate((hp,Jp))
    return hp,Jp

def convert_corr(si,sisj,convertTo='11',concat=False):
    """
    Convert single spin means and pairwise correlations between {0,1} and {-1,1} formulations.

    Params:
    -------
    si (ndarray)
    sisj (ndarray)
    convertTo (str,'11')
        '11' will convert {0,1} formulation to +/-1 and '01' will convert +/-1 formulation to {0,1}
    concat (bool=False)
        If True, return concatenation of means and pairwise correlations.

    Returns:
    --------
    si
        Converted to appropriate basis
    sisj
        converted to appropriate basis
    """
    if convertTo=='11':
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = 4*sisj[k] - 2*si[i] - 2*si[j] + 1
                k += 1
        newsi = si*2-1
    else:
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = ( sisj[k] + si[i] + si[j] + 1 )/4.
                k += 1
        newsi = (si+1)/2
    if concat:
        return np.concatenate((newsi,newsisj))
    return newsi,newsisj

def state_probs(v,allstates=None,weights=None,normalized=True):
    """
    Get probability of unique states. There is an option to allow for weights counting of the words.
    
    Params:
    -------
    states (ndarray)
        (n_samples,n_dim)
    weights (vector)
    normalized (bool=True)
        Return probability distribution instead of frequency count
    
    Returns:
    --------
    freq (ndarray)
        Vector of the probabilities of each state
    allstates (ndarray)
        All unique states found in the data.
    """
    if v.ndim==1:
        v = v[:,None]
    n = v.shape[1]
    j = 0
    return_all_states = False

    if allstates is None:
        allstates = v[unique_rows(v)]
        uniqIx = unique_rows(v,return_inverse=True)
        freq = np.bincount( uniqIx )
        return_all_states = True
    else:
        if weights is None:
            weights = np.ones((v.shape[0]))
        
        freq = np.zeros(allstates.shape[0])
        for vote in allstates:
            ix = ( vote==v ).sum(1)==n
            freq[j] = (ix*weights).sum()
            j+=1
        if np.isclose(np.sum(freq),np.sum(weights))==0:
            import warnings
            warnings.warn("States not found in given list of all states.")
    if normalized:
        freq = freq.astype(float)/np.sum(freq)

    if return_all_states:
        return freq,allstates
    return freq

    
# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_pseudo_ising_helpers(N):
    """
    Define helper functions for using Pseudo method on fully connected Ising model.

    Parameters
    ----------
    N : int
        System size.

    Returns
    -------
    get_multipliers_r, calc_observables_r 
    """
    @jit(nopython=True)
    def get_multipliers_r(r,multipliers):
        """
        Return the parameters relevant for calculating the conditional probability of spin r.

        Parameters
        ----------
        r : int
        multipliers : ndarray

        Returns
        -------
        multipliers
        """
        ix = np.arange(N)
        ix[0] = r  # index for local field
        couplingcounter = N
        ixcounter = 1
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                if i==r or j==r:
                    ix[ixcounter] = couplingcounter  # indices for couplings
                    ixcounter += 1
                couplingcounter += 1
        
        return multipliers[ix]

    @jit(nopython=True)
    def calc_observables_r(r,X):
        """
        Return the observables relevant for calculating the conditional probability of spin r.

        Parameters
        ----------
        r : int
        X : ndarray
            Data samples of dimensions (n_samples, n_dim).

        Returns
        -------
        observables
        """
        obs = np.zeros((X.shape[0],N))
        
        for rowix in xrange(X.shape[0]):
            ixcounter = 1
            obs[rowix,0] = X[rowix,r]
            
            for i in xrange(N-1):
                for j in xrange(i+1,N):
                    if i==r or j==r:
                        obs[rowix,ixcounter] = X[rowix,i]*X[rowix,j]  # indices for couplings
                        ixcounter += 1
        return obs

    return get_multipliers_r,calc_observables_r 

def define_ising_helpers_functions():
    """
    Functions for plugging into solvers for +/-1 Ising model with fields h_i and couplings J_ij.

    Returns
    -------
    calc_e
    calc_observables
    mch_approximation
    """
    @jit(nopython=True,cache=True)
    def fast_sum(J,s):
        """Helper function for calculating energy in calc_e(). Iterates couplings J."""
        e = np.zeros((s.shape[0]))
        for n in xrange(s.shape[0]):
            k = 0
            for i in xrange(s.shape[1]-1):
                for j in xrange(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e
    
    def calc_e(s,params):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}
        params : ndarray
            (h,J) vector
        """
        e = -fast_sum(params[s.shape[1]:],s)
        e -= np.dot(s,params[:s.shape[1]])
        return e

    def mch_approximation( samples, dlamda ):
        """Function for making MCH approximation step for Ising model."""
        dE = calc_e(samples,dlamda)
        ZFraction = len(dE) / np.exp(logsumexp(-dE))
        predsisj = pair_corr( samples, weights=np.exp(-dE)/len(dE),concat=True ) * ZFraction  
        assert not (np.any(predsisj<-1.00000001) or
            np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(),
                                                                                               predsisj.max())
        return predsisj
    
    @jit(nopython=True)
    def calc_observables(samples):
        """Observables for Ising model."""
        n = samples.shape[1]
        obs = np.zeros((samples.shape[0],n+n*(n-1)//2))
        
        k = 0
        for i in xrange(n):
            obs[:,i] = samples[:,i]
            for j in xrange(i+1,n):
                obs[:,n+k] = samples[:,i]*samples[:,j]
                k += 1
        return obs
    return calc_e,calc_observables,mch_approximation

def define_sising_helpers_functions():
    """
    Functions for plugging into solvers for +/-1 Ising model with couplings J_ij and no fields.

    Returns
    -------
    calc_e
    calc_observables
    mch_approximation
    """
    @jit(nopython=True,cache=True)
    def fast_sum(J,s):
        """Helper function for calculating energy in calc_e(). Iterates couplings J."""
        e = np.zeros((s.shape[0]))
        for n in xrange(s.shape[0]):
            k = 0
            for i in xrange(s.shape[1]-1):
                for j in xrange(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e
    
    @jit(nopython=True)
    def calc_e(s,params):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}
        params : ndarray
            (h,J) vector
        """
        return -fast_sum(params,s)

    def mch_approximation( samples, dlamda ):
        """Function for making MCH approximation step for symmetrized Ising model."""
        dE = calc_e(samples,dlamda)
        dE -= dE.min()
        ZFraction = 1. / np.mean(np.exp(-dE))
        predsisj = pair_corr( samples, weights=np.exp(-dE)/len(dE) )[1] * ZFraction  
        assert not (np.any(predsisj<-1.00000001) or
            np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(),
                                                                                               predsisj.max())
        return predsisj
    
    @jit(nopython=True)
    def calc_observables(samples):
        """Observables for symmetrized Ising model."""
        n = samples.shape[1]
        obs = np.zeros((samples.shape[0],n*(n-1)//2))
        
        k = 0
        for i in xrange(n):
            for j in xrange(i+1,n):
                obs[:,k] = samples[:,i]*samples[:,j]
                k += 1
        return obs
    return calc_e,calc_observables,mch_approximation


@jit(nopython=True)
def adj(s,n_random_neighbors=0):
    """
    Return one-flip neighbors and a set of random neighbors. This is written to be used with
    the solvers.MPF class. Use adj_sym() if symmetric spins in {-1,1} are needed.
    
    NOTE: For random neighbors, there is no check to make sure neighbors don't repeat but this
    shouldn't be a problem as long as state space is large enough.

    Parameters
    ----------
    s : ndarray
        State whose neighbors are found. One-dimensional vector of spins.
    n_random_neighbors : int,0
        If >0, return this many random neighbors. Neighbors are just random states, but they are
        called "neighbors" because of the terminology in MPF.

    Returns
    -------
    neighbors : ndarray
        Each row is a neighbor. s.size + n_random_neighbors are returned.
    """
    neighbors = np.zeros((s.size+n_random_neighbors,s.size))
    for i in xrange(s.size):
        s[i] = 1-s[i]
        neighbors[i] = s.copy()
        s[i] = 1-s[i]
    if n_random_neighbors:
        for i in xrange(n_random_neighbors):
            match = True
            while match:
                newneighbor = (np.random.rand(s.size)<.5)*1.
                # Make sure neighbor is not the same as the given state.
                if (newneighbor!=s).any():
                    match=False
            neighbors[i+s.size] = newneighbor
    return neighbors

@jit(nopython=True)
def adj_sym(s,n_random_neighbors=False):
    """
    Symmetric version of adj() where spins are in {-1,1}.
    """
    neighbors = np.zeros((s.size+n_random_neighbors,s.size))
    for i in xrange(s.size):
        s[i] = -1*s[i]
        neighbors[i] = s.copy()
        s[i] = -1*s[i]
    if n_random_neighbors:
        for i in xrange(n_random_neighbors):
            match=True
            while match:
                newneighbor=(np.random.rand(s.size)<.5)*2.-1
                if np.sum(newneighbor*s)!=s.size:
                    match=False
            neighbors[i+s.size]=newneighbor
    return neighbors

def calc_de(s,i):
    """
    Calculate the derivative of the energy wrt parameters given the state and
    index of the parameter. In this case, the parameters are the concatenated
    vector of {h_i,J_ij}.
    """
    if i<s.shape[1]:
        return -s[:,i]
    else:
        i-=s.shape[1]
        i,j=sub_to_pair_idx(i,s.shape[1])
        return -s[:,i]*s[:,j]
