"""Indexing, state generation, and basic matrix helpers.

Internal module split out of the original coniii.utils. Public names
are re-exported by :mod:`coniii.utils`.
"""
import numpy as np
from numba import jit, njit
from scipy.special import binom
from scipy.spatial.distance import squareform

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

    # Collapse each row to a single void scalar and ravel to 1-D. The ravel
    # matters under numpy >= 2.0, where np.unique(return_inverse=True) returns
    # the inverse with the same shape as its input; without a 1-D input the
    # inverse comes back 2-D and breaks downstream np.bincount calls.
    b = np.ascontiguousarray(mat).view(
        np.dtype((np.void, mat.dtype.itemsize * mat.shape[1]))).ravel()
    if not return_inverse:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx = np.unique(b, return_inverse=True)

    return idx


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

