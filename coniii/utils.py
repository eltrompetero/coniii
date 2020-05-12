# ========================================================================================================= #
# Miscellaneous functions used for various computations.
# Author : Edward Lee, edlee@santafe.edu
#
# MIT License
# 
# Copyright (c) 2020 Edward D. Lee, Bryan C. Daniels
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
from scipy.special import binom
from warnings import warn
NUMERALS = '0123456789'
ALPHNUM = '0123456789ABCDEFGHJIKLMNOPQRSTUVWXYZ'


@njit(cache=True)
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
        Number of spins.
    sym : bool
        If true, return states in {-1,1} basis.

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
        Dimensions (n_samples, n_dim).
    k : int
        Order of correlation function <s_{i_1} * s_{i_2} * ... * s_{i_k}>.
    weights : np.ndarray, None : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one.
    exclude_empty : bool, False
        When using with {-1,1} basis, you can leave entries with 0 and those will not be
        counted for any pair. If True, the weights option doesn't do anything.

    Returns
    -------
    ndarray
        Kth order correlations <s_{i_1} * s_{i_2} * ... * s_{i_k}>.
    """
    
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

def xbin_states(n, sym=False):
    """Generator for iterating through all possible binary states.

    Parameters
    ----------
    n : int
        Number of spins.
    sym : bool
        If true, return states in {-1,1} basis.

    Returns
    -------
    generator
    """

    assert n>0, "n cannot be <0"
    
    def v():
        for i in range(2**n):
            if sym is False:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')
            else:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')*2-1

    return v()

@njit
def xpotts_states(n, k):
    """Generator for iterating through all states for Potts model with k distinct states.
    This is a faster version of calling xbin_states(n, False) except with strings returned
    as elements instead of integers.

    Parameters
    ----------
    n : int
        Number of spins.
    k : int
        Number of distinct states. These are labeled by integers starting from 0 and must
        be <=36.

    Returns
    -------
    generator
    """

    assert n>0, "n cannot be <0"
    assert k>=2, "k cannot be <2"
   
    for i in range(k**n):
        state = base_repr(i, k)
        yield ['0']*(n-len(state)) + state

@njit
def base_repr(i, base):
    """Return decimal number in given base as list.
    
    Parameters
    ----------
    i : int
    base : int

    Returns
    -------
    list
    """

    assert i>=0 and base>=2
    
    if i==0:
        return ['0']

    if base<=10:
        return _small_base(i, base)

    assert base<=36
    return _large_base(i, base)

@njit
def _small_base(i, base):
    rep = []
    exponent = int(np.log(i)/np.log(base))
    term = int(i/base**exponent)
    # handle problematically rounded cases
    if term==base:
        exponent += 1
        term = 1
    rep.append(NUMERALS[term])
    i -= term*base**exponent
    
    exponent -= 1
    while exponent>=0:
        baseToExp = base**exponent
        if baseToExp>i:
            rep.append('0')
        else:
            term = int(i/baseToExp)
            rep.append(NUMERALS[term])
            i -= term*baseToExp
        exponent -= 1
    return rep
  
@njit
def _large_base(i, base):
    ALPHANUM = '0123456789ABCDEFGHJIKLMNOPQRSTUVWXYZ'
    rep = []
    exponent = int(np.log(i)/np.log(base))
    term = int(i/base**exponent)
    # handle problematically rounded cases
    if term==base:
        exponent += 1
        term = 1
    rep.append(ALPHANUM[term])
    i -= term*base**exponent
    
    exponent -= 1
    while exponent>=0:
        baseToExp = base**exponent
        if baseToExp>i:
            rep.append('0')
        else:
            term = int(i/baseToExp)
            rep.append(ALPHANUM[term])
            i -= term*baseToExp
        exponent -= 1
    return rep

def convert_params(h, J, convert_to, concat=False):
    """Convert Ising model fields and couplings from {0,1} basis to {-1,1} and vice versa.

    Parameters
    ----------
    h : ndarray
        Fields.
    J : ndarray
        Couplings.
    convert_to : str
        Either '01' or '11'.
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
        Raveled index to unravel.
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
    n : int, 2
    """
    
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
    """Get probability of unique states. There is an option to allow for weighted
    counting.
    
    Parameters
    ----------
    states : ndarray
        Sample of states on which to extract probabilities of unique configurations with
        dimensions (n_samples,n_dimension).
    allstates : ndarray, None
        Unique configurations to look for with dimensions (n_samples, n_dimension).
    weights : vector, None
        For weighted counting of each state given in allstate kwarg.
    normalized : bool, True
        If True, return probability distribution instead of frequency count.
    
    Returns
    -------
    ndarray
        Vector of the probabilities of each state.
    ndarray
        All unique states found in the data. Each state is a row. Only returned if
        allstates kwarg is not provided.
    """

    if v.ndim==1:
        v = v[:,None]
    n = v.shape[1]
    j = 0
    return_all_states = False  # switch to keep track of whether or not allstates were given

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

def replace_diag(mat, newdiag):
    """Replace diagonal entries of square matrix.

    Parameters
    ----------
    mat : ndarray
    newdiag : ndarray

    Returns
    -------
    ndarray
    """

    if newdiag.ndim>1: 
        raise Exception("newdiag should be 1-dimensional")
    if not (mat.shape[0]==mat.shape[1]==newdiag.size):
        raise Exception("Incorrect dimensions.")
    return mat - np.diag(mat.diagonal()) + np.diag(newdiag)

def zero_diag(mat):
    """Replace diagonal entries of square matrix with zeros.

    Parameters
    ----------
    mat : ndarray

    Returns
    -------
    ndarray
    """

    return replace_diag(mat, np.zeros(mat.shape[0]))



# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_pseudo_ising_helper_functions(N):
    """Define helper functions for using Pseudo method on Ising model.

    Parameters
    ----------
    N : int
        System size.

    Returns
    -------
    function
        get_multipliers_r
    function
        calc_observables_r 
    """

    @njit
    def get_multipliers_r(r, multipliers, N=N):
        """Return r's field and all couplings to spin r.

        Parameters
        ----------
        r : int
        multipliers : ndarray
            All fields and couplings concatenated together.

        Returns
        -------
        ndarray
            Relevant multipliers.
        list
            Index of where multipliers appear in full multipliers array.
        """
        
        ix = [r] 
        multipliersr = np.zeros(N)
        multipliersr[0] = multipliers[r]  # first entry is the biasing field

        # fill in the couplings
        ixcounter = 1
        for i in range(N):
            if i!=r:
                if i<r:
                    ix.append( sub_to_ind(N, i, r) + N )
                    multipliersr[ixcounter] = multipliers[ix[ixcounter]]
                else:
                    ix.append( sub_to_ind(N, r, i) + N )
                    multipliersr[ixcounter] = multipliers[ix[ixcounter]]
                ixcounter += 1
        return multipliersr, ix

    @njit
    def calc_observables_r(r, X, N=N):
        """Return the observables relevant for calculating the conditional probability of
        spin r.

        Parameters
        ----------
        r : int
            Spin index.
        X : ndarray
            Data samples of dimensions (n_samples, n_dim).

        Returns
        -------
        ndarray
            observables
        """

        obs = np.zeros((X.shape[0],N))
        
        for rowix in range(X.shape[0]):
            ixcounter = 1
            obs[rowix,0] = X[rowix,r]
            
            for i in range(N-1):
                for j in range(i+1,N):
                    if i==r or j==r:
                        obs[rowix,ixcounter] = X[rowix,i]*X[rowix,j]
                        ixcounter += 1
        return obs

    return get_multipliers_r, calc_observables_r 

def define_pseudo_potts_helper_functions(n, k):
    """Define helper functions for using Pseudo method on Potts model with simple form for
    couplings that are only nonzero when the spins are occupying the same state.

    Parameters
    ----------
    n : int
        System size.
    k : int
        Number of possible configurations in Potts model.

    Returns
    -------
    function
        get_multipliers_r
    function
        calc_observables_r 
    """

    assert n>1
    assert k>1

    @njit
    def get_multipliers_r(r, multipliers, n=n, k=k):
        """Return r's field and all couplings to spin r.

        Parameters
        ----------
        r : int
        multipliers : ndarray
            All fields and couplings concatenated together.

        Returns
        -------
        ndarray
            Relevant multipliers.
        list
            Index of where multipliers appear in full multipliers array.
        """
        
        ix = [r+n*i for i in range(k)]
        multipliersr = np.zeros(k-1+n)
        for i in range(k):
            multipliersr[i] = multipliers[r+n*i]

        # fill in the couplings
        ixcounter = k
        for i in range(n):
            if i!=r:
                if i<r:
                    ix.append( sub_to_ind(n, i, r) + k*n )
                    multipliersr[ixcounter] = multipliers[ix[ixcounter]]
                else:
                    ix.append( sub_to_ind(n, r, i) + k*n )
                    multipliersr[ixcounter] = multipliers[ix[ixcounter]]
                ixcounter += 1
        return multipliersr, ix

    @njit
    def calc_observables_r(r, X, n=n, k=k):
        """Return the observables relevant for calculating the conditional probability of
        spin r.

        Parameters
        ----------
        r : int
            Spin index.
        X : ndarray
            Data samples of dimensions (n_samples, n_dim).

        Returns
        -------
        ndarray
            observables
        list of ndarray
            observables if spin r were to occupy all other possible states
        ndarray
            Each col details the occupied by spin r in each array of the previous return
            value, i.e., the first col of this array tells me what r has been changed to
            in the first array in the above list.
        """

        obs = np.zeros((X.shape[0],k-1+n), dtype=np.int8)
        # keep another copy of observables where the spin iterates thru all other possible states
        otherobs = [np.zeros((X.shape[0],k-1+n), dtype=np.int8)
                    for i in range(k-1)]
        # note the hypothetical states occupied by spin r. this makes it easier to keep track of things later
        otherstates = np.zeros((X.shape[0],k-1), dtype=np.int8)
        
        # for each data sample in X
        for rowix in range(X.shape[0]):
            counter = 0
            # record state of spin r in obs and hypothetical scenarios when it is another state in otherobs
            for i in range(k):
                if X[rowix,r]==i:
                    obs[rowix,i] = 1
                else:
                    otherobs[counter][rowix,i] = 1
                    otherstates[rowix,counter] = i
                    counter += 1
            ixcounter = k
            
            for i in range(n-1):
                for j in range(i+1,n):
                    if i==r:
                        obs[rowix,ixcounter] = X[rowix,i]==X[rowix,j]
                        kcounter = 0
                        for state in range(k):
                            if state!=X[rowix,r]:
                                otherobs[kcounter][rowix,ixcounter] = X[rowix,j]==state
                                kcounter += 1
                        ixcounter += 1
                    elif j==r:
                        obs[rowix,ixcounter] = X[rowix,i]==X[rowix,j]
                        kcounter = 0
                        for state in range(k):
                            if state!=X[rowix,r]:
                                otherobs[kcounter][rowix,ixcounter] = X[rowix,i]==state
                                kcounter += 1
                        ixcounter += 1
        return obs, otherobs, otherstates

    return get_multipliers_r, calc_observables_r 

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
    function
        calc_e
    function
        calc_observables
    function
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

def define_triplet_helper_functions():
    @njit
    def calc_observables(X):
        """Triplet order model consists of constraining all the correlations up to third order."""
        
        n = X.shape[1]
        Y = np.zeros((len(X), n+n*(n-1)//2+n*(n-1)*(n-2)//6))
        
        # average orientation (magnetization)
        counter = 0
        for i in range(n):
            Y[:,counter] = X[:,i]
            counter += 1
        
        # pairwise correlations
        for i in range(n-1):
            for j in range(i+1, n):
                Y[:,counter] = X[:,i]*X[:,j]
                counter += 1
                
        # triplet correlations
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    Y[:,counter] = X[:,i]*X[:,j]*X[:,k]
                    counter += 1
        return Y

    def calc_e(X, multipliers):
        return -calc_observables(X).dot(multipliers)

    return calc_e, calc_observables

def define_ternary_helper_functions():
    @njit
    def calc_observables(X):
        """Triplet order model consists of constraining all the correlations up to third
        order.
        """
        
        n = X.shape[1]
        Y = np.zeros((len(X), n*3+n*(n-1)//2))
        
        # average orientation (magnetization)
        counter = 0
        for i in range(3*n):
            Y[:,counter] = X[:,i]
            counter += 1
        
        # pairwise correlations
        for i in range(n-1):
            for j in range(i+1, n):
                Y[:,counter] = X[:,i]*X[:,j]
                counter += 1
                
        return Y

    def calc_e(X, multipliers):
        return -calc_observables(X).dot(multipliers)

    return calc_e, calc_observables

def define_potts_helper_functions(k):
    """Helper functions for calculating quantities in k-state Potts model.

    Parameters
    ----------
    k : int 
        Number of possible states.

    Returns
    -------
    function
        calc_e
    function
        calc_observables
    function
        mch_approximation
    """

    @njit
    def calc_observables(X, k=k):
        """
        Parameters
        ----------
        X : ndarray of dtype np.int64
            Dimensions (n_samples, n_spins).

        Returns
        -------
        ndarray
            Dimensions (n_samples, n_observables).
        """

        n = X.shape[1]
        Y = np.zeros((len(X), n*k+n*(n-1)//2), dtype=np.int8)
        
        # average orientation (magnetization)
        # note that fields for the third state are often set to 0
        counter = 0
        for i in range(k):
            for j in range(n):
                Y[:,counter] = X[:,j]==i
                counter += 1
        
        # pairwise correlations
        for i in range(n-1):
            for j in range(i+1, n):
                Y[:,counter] = X[:,i]==X[:,j]
                counter += 1
                
        return Y

    def calc_e(X, multipliers, k=k, calc_observables=calc_observables):
        """
        Parameters
        ----------
        X : ndarray of dtype np.int64
            Dimensions (n_samples, n_spins).
        multipliers : ndarray of dtype np.float64

        Returns
        -------
        ndarray
            Energies of each observable.
        """

        return -calc_observables(X, k).dot(multipliers)

    def mch_approximation(sample, dlamda, calc_e=calc_e):
        """Function for making MCH approximation step for Potts model.
        
        Parameters
        ----------
        sample : ndarray
            Of dimensions (n_sample, n_spins).
        dlamda : ndarray
            Change in parameters.
        
        Returns
        -------
        ndarray
            Predicted correlations.
        """

        dE = calc_e(sample, dlamda)
        ZFraction = len(dE) / np.exp(logsumexp(-dE))
        predsisj = (np.exp(-dE[:,None]) / len(dE) * calc_observables(sample)).sum(0) * ZFraction  
        assert not ((predsisj<0).any() or
                    (predsisj>(1+1e-10)).any()),"Predicted values are beyond limits, (%E,%E)"%(predsisj.min(),
                                                                                               predsisj.max())
        return predsisj

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

def vec2mat(multipliers, separate_fields=False):
    """Convert vector of parameters containing fields and couplings to a matrix where the
    diagonal elements are the fields and the remaining elements are the couplings. Fields
    can be returned separately with the separate_fields keyword argument.

    This is specific to the Ising model.
    
    Parameters
    ----------
    multipliers : ndarray
        Vector of fields and couplings.
    separate_fields : bool, False
    
    Returns
    -------
    ndarray
        n x n matrix. Diagonal elements are fields *unless* separate_fields keyword
        argument is True, in which case the diagonal elements are 0.
    ndarray (optional)
        Fields if separate_fields keyword argument is True.
    """

    n = (np.sqrt(1+8*multipliers.size)-1)//2
    assert (n%1)==0, "Must be n fields and (n choose 2) couplings."
    n = int(n)

    if separate_fields:
        return multipliers[:n], squareform(multipliers[n:])
    return replace_diag(squareform(multipliers[n:]), multipliers[:n])

def mat2vec(multipliers):
    """Convert matrix form of Ising parameters to a vector. 

    This is specific to the Ising model.
    
    Parameters
    ----------
    multipliers : ndarray
        Matrix of couplings with diagonal elements as fields.
    
    Returns
    -------
    ndarray
        Vector of fields and couplings, respectively.
    """

    return np.concatenate([multipliers.diagonal(), squareform(zero_diag(multipliers))])
