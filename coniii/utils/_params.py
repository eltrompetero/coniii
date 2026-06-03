"""Parameter conversion and maxent helper-function factories.

Internal module split out of the original coniii.utils. Public names
are re-exported by :mod:`coniii.utils`.
"""
import numpy as np
from numba import jit, njit
from itertools import combinations
from scipy.special import logsumexp, binom
from scipy.spatial.distance import squareform

from ._indexing import sub_to_ind, bin_states, unravel_index, multinomial
from ._correlations import pair_corr, calc_overlap

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
    def fast_sum(J, s):
        """Helper function for calculating energy in calc_e(). Iterates couplings J."""
        e = np.zeros(s.shape[0])
        for n in range(s.shape[0]):
            k = 0
            for i in range(s.shape[1]-1):
                for j in range(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e

    @njit
    def calc_e(s, params):
        """
        Parameters
        ----------
        s : 2D ndarray of ints
            state either {0,1} or {+/-1}; any integer width
        params : ndarray
            (h, J) vector

        Returns
        -------
        E : ndarray
            Energies of all given states.
        """

        e = -fast_sum(params[s.shape[1]:],s)
        e -= np.sum(s*params[:s.shape[1]],1)
        return e
    
    def mch_approximation(samples, dlamda):
        """Function for making MCH approximation step for Ising model."""
        dE = calc_e(samples, dlamda)
        ZFraction = len(dE) / np.exp(logsumexp(-dE))
        predsisj = pair_corr(samples, weights=np.exp(-dE)/len(dE), concat=True) * ZFraction  
        assert not (np.any(predsisj < -1.00000001) or
            np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(),
                                                                                               predsisj.max())
        return predsisj
    
    @njit(cache=True)
    def calc_observables(samples):
        """Observables for Ising model."""
        n = samples.shape[1]
        obs = np.zeros((samples.shape[0], n+n*(n-1)//2))
        
        k = 0
        for i in range(n):
            obs[:,i] = samples[:,i]
            for j in range(i+1,n):
                obs[:,n+k] = samples[:,i] * samples[:,j]
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
    
    @njit
    def calc_e(s, params):
        """
        Parameters
        ----------
        s : 2D ndarray
            state either {0,1} or {+/-1}; any integer width
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

