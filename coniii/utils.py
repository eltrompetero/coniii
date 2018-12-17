# ========================================================================================================= #
# Miscellaneous functions used for various computations.
# Author : Edward Lee, edlee@alumni.princeton.edu
#
# MIT License
# 
# Copyright (c) 2017 Edward D. Lee, Bryan C. Daniels
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
# ========================================================================================================= #
import numpy as np
from numba import jit,njit
from scipy.special import logsumexp
from itertools import combinations
from scipy.spatial.distance import squareform


@njit(nogil=True, cache=True)
def sub_to_ind(n, i, j):
    """Convert pair of coordinates of a symmetric square array into consecutive index of
    flattened upper triangle. This is slimmed down so it won't throw errors like if i>n or
    j>n or if they're negative. Only checking for if the returned index is negative which
    could be problematic with wrapped indices.
    
    Parameters
    ----------
    n : int
        Dimension of square array
    i,j : int
        coordinates

    Returns
    -------
    int
    """

    if i<j:
        k = 0
        for l in range(1,i+2):
            k += n-l
        assert k>=0
        return k-n+j
    elif i>j:
        k = 0
        for l in range(1,j+2):
            k += n-l
        assert k>=0
        return k-n+i
    else:
        raise Exception("Indices cannot be the same.")

@njit(cache=True)
def ind_to_sub(n, ix):
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
    for i in range(n-1):
        for j in range(i+1,n):
            if k==ix:
                return (i,j)
            k += 1
 
def unique_rows(mat, return_inverse=False):
    """Return unique rows indices of a numeric numpy array.

    Parameters
    ----------
    mat : ndarray
    return_inverse : bool
        If True, return inverse that returns back indices of unique array that would
        return the original array 

    Returns
    -------
    u : ndarray
        Unique elements of matrix.
    idx : ndarray
        row indices of given mat that will give unique array
    """

    b = np.ascontiguousarray(mat).view(np.dtype((np.void, mat.dtype.itemsize * mat.shape[1])))
    if not return_inverse:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx = np.unique(b, return_inverse=True)
    
    return idx

def calc_overlap(sample,ignore_zeros=False):
    """<si_a si_b> between all pairs of replicas a and b

    Params:
    -------
    sample
    ignore_zeros (bool=False)
        Instead of normalizing by the number of spins, normalize by the minimum number of
        nonzero spins.
    """
    overlap = sample.dot(sample.T)
    if ignore_zeros:
        countZeros = np.zeros((len(sample),len(sample),2))
        countZeros[:,:,0] = (sample==0).sum(1)[:,None]
        countZeros[:,:,1] = (sample==0).sum(1)[None,:]
        return overlap / (sample.shape[1]-countZeros.max(2))
    return overlap / sample.shape[1]

def pair_corr(X,
              weights=None,
              concat=False,
              exclude_empty=False,
              subtract_mean=False,
              laplace_count=False):
    """Calculate averages and pairwise correlations of spins.

    Parameters
    ----------
    X : ndarray
        Dimensions (n_samples,n_dim).
    weights : float or np.ndarray or twople, None
        If an array is passed, it must be the length of the data and each data point will
        be given the corresponding weight. Otherwise, the two element tuple should contain
        the normalization for each mean and each pairwise correlation, in that order. In
        other words, the first array should be length {s_i} and the second length
        {si*s_j}.
    concat : bool, False
        Return means concatenated with the pairwise correlations into one array.
    exclude_empty : bool, False
        When using with {-1,1} basis, you can leave entries with 0 and those will not be
        counted for any pair. If True, the weights option doesn't do anything.
    subtract_mean : bool, False
        If True, return pairwise correlations with product of individual means subtracted.
    laplace_count : 


    Returns
    -------
    twople
        (si,sisj) or np.concatenate((si,sisj))
    """

    assert frozenset(np.unique(X))<=frozenset([-1,0,1])
    S, N = X.shape
    
    if exclude_empty and not laplace_count:
        # count all nonzero entries for every pair
        weights = 1/(X!=0).sum(0), 1./( (X!=0).astype(int).T.dot(X!=0)[np.triu_indices(X.shape[1],k=1)] )
    elif exclude_empty and laplace_count:
        weights = ( 1/((X!=0).sum(0)+2),
                    1./( (X!=0).astype(int).T.dot(X!=0)[np.triu_indices(X.shape[1],k=1)] + 4 ) )
    elif weights is None:
        # for taking simple average
        weights = np.ones(len(X))/len(X)
    elif type(weights) is tuple:
        assert len(weights[0])==X.shape[1]
        assert len(weights[1])==(X.shape[1]*(X.shape[1]-1)//2)
    elif type(weights) is np.ndarray:
        assert len(weights)==len(X)
    else:
        weights = np.zeros(len(X))+weights
    
    # Calculate pairwise correlations depending on whether or not exclude_empty was set or not.
    if type(weights) is tuple:
        si = X.sum(0) * weights[0]
        sisj = (X.T.dot(X))[np.triu_indices(X.shape[1],k=1)] * weights[1]
    else:
        si = (X*weights[:,None]).sum(0)
        sisj = (X.T.dot(X*weights[:,None]))[np.triu_indices(X.shape[1],k=1)]

    if subtract_mean:
        sisj = np.array([sisj[i]-si[ix[0]]*si[ix[1]] for i,ix in enumerate(combinations(list(range(N)),2))])
    
    if concat:
        return np.concatenate((si,sisj))
    return si, sisj

def bin_states(n, sym=False):
    """Generate all possible binary spin states. 
    
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
    
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype(int)

    if sym is False:
        return v
    return v*2-1

def k_corr(X, k,
           weights=None,
           exclude_empty=False):
    """Calculate kth order correlations of spins.

    Parameters
    ----------
    X : ndarray
        Dimensions (n_samples,n_dim).
    k : int
        Order of correlation <s_{i_1} * s_{i_2} * ... * s_{i_k}>.
    weights : np.ndarray,None : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one
    exclude_empty : bool,False
        When using with {-1,1} basis, you can leave entries with 0 and those will not be
        counted for any pair. If True, the weights option doesn't do anything.

    Returns
    -------
    kcorr : ndarray
        <s_{i_1} * s_{i_2} * ... * s_{i_k}>.
    """
    
    from scipy.special import binom
    assert frozenset(np.unique(X))<=frozenset([-1,0,1])
    S, N = X.shape
    kcorr = np.zeros(int(binom(N,k)))
    
    if exclude_empty:
        for counter,ijk in enumerate(combinations(range(N), k)):
            p = np.prod(X[:,ijk], axis=1)
            kcorr[counter] = p[p!=0].mean()
    
    if weights is None:
        weights = np.ones(S)/S
    for counter,ijk in enumerate(combinations(range(N), k)):
        kcorr[counter] = np.prod(X[:,ijk], axis=1).dot(weights)
    return kcorr

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
        for i in range(2**n):
            if sym is False:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')
            else:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')*2-1

    return v()

def convert_params(h, J, convert_to, concat=False):
    """Convert Ising model fields and couplings from {0,1} basis to {-1,1} and vice versa.

    Parameters
    ----------
    h : ndarray
    J : ndarray
    convert_to : str
        '01' or '11'
    concat : bool, False
        If True, return a vector concatenating fields and couplings.
    
    Returns
    -------
    ndarray
        Mean bias h vector. Concatenated vector of h and J if concat is True.
    ndarray, optional
        Vector of J.
    """

    if len(J.shape)!=2:
        Jmat = squareform(J)
    else:
        Jmat = J
        J = squareform(J)
    
    if convert_to=='11':
        # Convert from 0,1 to -/+1
        Jp = J/4.
        hp = h/2 + np.sum(Jmat,1)/4.
    elif convert_to=='01':
        # Convert from -/+1 to 0,1
        hp = 2.*(h - np.sum(Jmat,1))
        Jp = J*4.
    else:
        raise Exception("Invalid choice for convert_to. Must be '01' or '11'.")

    if concat:
        return np.concatenate((hp, Jp))
    return hp, Jp

def ising_convert_params(oparams, convert_to, concat=False):
    """General conversion of parameters from 01 to 11 basis.

    Take set of Ising model parameters up to nth order interactions in either {0,1} or
    {-1,1} basis and convert to other basis.

    Parameters
    ----------
    oparams : tuple of lists
        Tuple of lists of interactions between spins starting with the lowest order
        interactions. Each list should consist of all interactions of that order such that
        the length of each list should be binomial(n,i) for all i starting with i>=1.
    convert_to : str
    concat : bool,False

    Returns
    -------
    params : tuple of lists or list
        New parameters in order of lowest to highest order interactions to mean biases.
        Can all be concatenated together if concat switch is True.
    """
    
    from scipy.special import binom
    oparams = oparams[::-1]
    n = len(oparams[-1])
    params = [np.zeros(int(binom(n,i))) for i in range(len(oparams),0,-1)]
    
    if convert_to=='01':
        # basically need to expand polynomials to all lower order terms
        # start at the highest order terms
        for counter,order in enumerate(range(len(oparams),0,-1)):
            # iterate through all combinations of indices of order
            for ijkcounter,ijk in enumerate(combinations(range(n), order)):
                ijkcoeff = oparams[counter][ijkcounter]

                # same order term is only multiplied by powers of two
                params[counter][ijkcounter] += 2**order * ijkcoeff
                # break this term down to lower order terms in the new basis
                for subcounter,suborder in enumerate(range(order-1,0,-1)):
                    for subijk in combinations(ijk, suborder):
                        ix = unravel_index(subijk, n)
                        params[subcounter+counter+1][ix] += ijkcoeff * 2**suborder * (-1)**(order-suborder)
    elif convert_to=='11':
        # basically need to expand polynomials to all lower order terms
        # start at the highest order terms
        for counter,order in enumerate(range(len(oparams),0,-1)):
            # iterate through all combinations of indices of order
            for ijkcounter,ijk in enumerate(combinations(range(n), order)):
                ijkcoeff = oparams[counter][ijkcounter]

                # same order term is only multiplied by powers of two
                params[counter][ijkcounter] += 2**-order * ijkcoeff
                # break this term down to lower order terms in the new basis
                for subcounter,suborder in enumerate(range(order-1,0,-1)):
                    for subijk in combinations(ijk, suborder):
                        ix = unravel_index(subijk, n)
                        params[subcounter+counter+1][ix] += ijkcoeff * 2**-order
    else:
        raise Exception("Invalid choice for convert_to. Must be either '01' or '11'.")

    if concat:
        return np.concatenate(params[::-1])
    return params[::-1]

def unravel_index(ijk, n):
    """Unravel multi-dimensional index to flattened index but specifically for
    multi-dimensional analog of an upper triangular array (lower triangle indices are not
    counted).

    Parameters
    ----------
    ijk : tuple
        Raveled index to unravel. These must be sorted increasing order.
    n : int
        System size.

    Returns
    -------
    ix : int
        Unraveled index.
    """
    
    if type(ijk) is int:
        return ijk
    if len(ijk)==1:
        return ijk[0]

    from scipy.special import binom
    assert (np.diff(ijk)>0).all()
    assert all([i<n for i in ijk])

    ix = sum([int(binom(n-1-i,len(ijk)-1)) for i in range(ijk[0])])
    for d in range(1, len(ijk)-1):
        if (ijk[d]-ijk[d-1])>1:
            ix += sum([int(binom(n-i-1,len(ijk)-d-1)) for i in range(ijk[d-1]+1, ijk[d])])
    ix += ijk[-1] -ijk[-2] -1
    return ix

def multinomial(*args):
    from scipy.special import factorial
    assert sum(args[1:])==args[0]
    return int(np.exp( np.log(factorial(args[0])) - sum([np.log(factorial(a)) for a in args[1:]]) ))

def _expand_binomial(a, b, n=2):
    """Expand a product of binomials that have the same coefficients given by a and b.
    E.g. (a*x0 + b) * (a*x1 + b) * ... * (a*xn + b)

    Parameters
    ----------
    a : float
    b : float
    n : int,2
    """
    
    from scipy.special import binom
    coeffs=[]
    for i in range(n+1):
        coeffs.extend( [a**(n-i)*b**i]*int(binom(n,i)) )
    return coeffs

def split_concat_params(p, n):
    """Split parameters for Ising model that have all been concatenated together into a
    single list into separate lists. Assumes that the parameters are increasing in order
    of interaction and that all parameters are present.
    
    Parameters
    ----------
    p : list-like
    
    Returns
    -------
    list of list-like
        Parameters increasing in order: (h, Jij, Kijk, ... ).
    """
    
    from scipy.special import binom
    splitp = []
    counter = 0
    i = 1
    while counter<len(p):
        splitp.append( p[counter:counter+int(binom(n,i))] )
        counter += int(binom(n,i))
        i += 1
    return splitp

def convert_corr(si, sisj, convert_to, concat=False, **kwargs):
    """Convert single spin means and pairwise correlations between {0,1} and {-1,1}
    formulations.

    Parameters
    ----------
    si : ndarray
        Individual means.
    sisj : ndarray
        Pairwise correlations.
    convert_to : str
        '11' will convert {0,1} formulation to +/-1 and '01' will convert +/-1 formulation
        to {0,1}
    concat : bool, False
        If True, return concatenation of means and pairwise correlations.

    Returns
    -------
    ndarray
        Averages <si>. Converted to appropriate basis. Returns concatenated vector <si>
        and <sisj> if concat is True.
    ndarray, optional
        Pairwise correlations <si*sj>. Converted to appropriate basis.
    """

    if 'convertTo' in kwargs.keys():
        from warnings import warn
        warn("convertTo kwarg is deprecated as of v1.1.2. Use convert_to instead.")
        convert_to = convertTo
    elif len(kwargs.keys())>0:
        raise TypeError("Unexpected keyword argument.")

    if convert_to=='11':
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = 4*sisj[k] - 2*si[i] - 2*si[j] + 1
                k += 1
        newsi = si*2-1
    elif convert_to=='01':
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = ( sisj[k] + si[i] + si[j] + 1 )/4.
                k += 1
        newsi = (si+1)/2
    else:
        raise Exception("Invalid value for convert_to. Must be either '01' or '11'.")

    if concat:
        return np.concatenate((newsi,newsisj))
    return newsi, newsisj

def state_probs(v, allstates=None, weights=None, normalized=True):
    """Get probability of unique states. There is an option to allow for weights counting
    of the words.
    
    Parameters
    ----------
    states : ndarray
        (n_samples,n_dim)
    allstates : ndarray, None
    weights : vector, None
    normalized : bool, True
        Return probability distribution instead of frequency count
    
    Returns
    -------
    ndarray
        Vector of the probabilities of each state.
    ndarray
        All unique states found in the data. Each state is a row.
    """

    if v.ndim==1:
        v = v[:,None]
    n = v.shape[1]
    j = 0
    return_all_states = False

    if allstates is None:
        allstates = v[unique_rows(v)]
        uniqIx = unique_rows(v, return_inverse=True)
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
        return freq, allstates
    return freq

    
# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_pseudo_ising_helpers(N):
    """Define helper functions for using Pseudo method on fully connected Ising model.

    Parameters
    ----------
    N : int
        System size.

    Returns
    -------
    get_multipliers_r, calc_observables_r 
    """

    @njit
    def get_multipliers_r(r, multipliers):
        """Return the parameters relevant for calculating the conditional probability of
        spin r.

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
        for i in range(N-1):
            for j in range(i+1,N):
                if i==r or j==r:
                    ix[ixcounter] = couplingcounter  # indices for couplings
                    ixcounter += 1
                couplingcounter += 1
        
        return multipliers[ix]

    @njit
    def calc_observables_r(r, X):
        """Return the observables relevant for calculating the conditional probability of
        spin r.

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
        
        for rowix in range(X.shape[0]):
            ixcounter = 1
            obs[rowix,0] = X[rowix,r]
            
            for i in range(N-1):
                for j in range(i+1,N):
                    if i==r or j==r:
                        obs[rowix,ixcounter] = X[rowix,i]*X[rowix,j]  # indices for couplings
                        ixcounter += 1
        return obs

    return get_multipliers_r,calc_observables_r 

def define_ising_helper_functions():
    """Functions for plugging into solvers for +/-1 Ising model with fields h_i and
    couplings J_ij.

    Returns
    -------
    function
        calc_e
    function
        calc_observables
    function
        mch_approximation
    """

    @njit(cache=True)
    def fast_sum(J,s):
        """Helper function for calculating energy in calc_e(). Iterates couplings J."""
        e = np.zeros((s.shape[0]))
        for n in range(s.shape[0]):
            k = 0
            for i in range(s.shape[1]-1):
                for j in range(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e

    @njit("float64[:](int64[:,:],float64[:])")
    def calc_e(s, params):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}
        params : ndarray
            (h,J) vector

        Returns
        -------
        E : ndarray
            Energies of all given states.
        """
        
        e = -fast_sum(params[s.shape[1]:],s)
        e -= np.sum(s*params[:s.shape[1]],1)
        return e
    
    def mch_approximation( samples, dlamda ):
        """Function for making MCH approximation step for Ising model."""
        dE = calc_e(samples, dlamda)
        ZFraction = len(dE) / np.exp(logsumexp(-dE))
        predsisj = pair_corr( samples, weights=np.exp(-dE)/len(dE),concat=True ) * ZFraction  
        assert not (np.any(predsisj<-1.00000001) or
            np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(),
                                                                                               predsisj.max())
        return predsisj
    
    @njit(cache=True)
    def calc_observables(samples):
        """Observables for Ising model."""
        n = samples.shape[1]
        obs = np.zeros((samples.shape[0],n+n*(n-1)//2))
        
        k = 0
        for i in range(n):
            obs[:,i] = samples[:,i]
            for j in range(i+1,n):
                obs[:,n+k] = samples[:,i]*samples[:,j]
                k += 1
        return obs
    return calc_e, calc_observables, mch_approximation

def define_ising_helper_functions_sym():
    """Functions for plugging into solvers for +/-1 Ising model with couplings J_ij and no
    fields.

    Returns
    -------
    calc_e
    calc_observables
    mch_approximation
    """
    @njit("float64[:](int64[:],float64[:,:])", cache=True)
    def fast_sum(J,s):
        """Helper function for calculating energy in calc_e(). Iterates couplings J."""
        e = np.zeros((s.shape[0]))
        for n in range(s.shape[0]):
            k = 0
            for i in range(s.shape[1]-1):
                for j in range(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e
    
    @njit("float64[:](int64[:,:],float64[:])")
    def calc_e(s, params):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}
        params : ndarray
            (h,J) vector

        Returns
        -------
        E : ndarray
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
    
    @njit
    def calc_observables(samples):
        """Observables for symmetrized Ising model."""
        n = samples.shape[1]
        obs = np.zeros((samples.shape[0],n*(n-1)//2))
        
        k = 0
        for i in range(n):
            for j in range(i+1,n):
                obs[:,k] = samples[:,i]*samples[:,j]
                k += 1
        return obs
    return calc_e, calc_observables, mch_approximation


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

def calc_de(s, i):
    """Calculate the derivative of the energy wrt parameters given the state and index of
    the parameter. In this case, the parameters are the concatenated vector of {h_i,J_ij}.

    Parameters
    ----------
    s : ndarray
         Two-dimensional vector of spins where each row is a state.
    i : int

    Returns
    -------
    dE : float
        Derivative of hamiltonian with respect to ith parameter, i.e. the corresponding
        observable.
    """
    
    assert s.ndim==2
    if i<s.shape[1]:
        return -s[:,i]
    else:
        i -= s.shape[1]
        i, j = ind_to_sub(s.shape[1], i)
        return -s[:,i] * s[:,j]
