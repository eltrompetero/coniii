"""Correlation and observable utilities.

Internal module split out of the original coniii.utils. Public names
are re-exported by :mod:`coniii.utils`.
"""
import warnings
import numpy as np
from numba import njit
from itertools import combinations
from scipy.special import logsumexp

from ._indexing import sub_to_ind, ind_to_sub, bin_states, unique_rows

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


def convert_corr(si, sisj, convert_to, concat=False):
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

