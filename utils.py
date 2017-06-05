from __future__ import division
import numpy as np
from numba import jit
from scipy.misc import logsumexp


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

def pair_corr(data,weights=None,concat=False,exclude_empty=False):
    """
    Calculate averages and pairwise correlations of spins.

    Params:
    -------
    data (ndarray)
        (n_samples,n_dim).
    weights (np.ndarray,None) : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one
    concat (bool,False)
        return concatenated means if true
    exclude_empty (bool,False)
        when using with {-1,1}, can leave entries with 0 and those will not be counted for any pair
        weights option doesn't do anything here

    Returns:
    --------
    (si,sisj) or np.concatenate((si,sisj))
    """
    S,N = data.shape
    sisj = np.zeros(N*(N-1)//2)
    
    if weights is None:
        weights = np.ones((data.shape[0]))/data.shape[0]

    if exclude_empty:
        assert np.array_equal( np.unique(data),np.array([-1,0,1]) ) or \
            np.array_equal( np.unique(data),np.array([-1,1]) ), "Only handles -1,1 data sets."
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = np.nansum(data[:,i]*data[:,j]) / np.nansum(np.logical_and(data[:,i]!=0,data[:,j]!=0))
                k+=1
        si = np.array([ np.nansum(col[col!=0]) / np.nansum(col!=0) for col in data.T ])
    else:
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = np.nansum(data[:,i]*data[:,j]*weights)
                k+=1
        si = np.nansum(data*weights[:,None],0)

    if concat:
        return np.concatenate((si,sisj))
    else:
        return si, sisj

def bin_states(n,sym=False):
    """
    Get all possible binary spin states. 
    
    Params:
    -------
    n (int)
        number of spins
    sym (bool)
        if true, return {-1,1} basis

    Returns:
    --------
    v (ndarray)
    """
    if n<0:
        raise Exception("n cannot be <0")
    if n>20:
        raise Exception("n is too large to enumerate all states.")
    
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype('uint8')

    if sym is False:
        return v
    else:
        return v*2.-1


# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_ising_mch_helpers():
    """
    Functions for plugging into GeneralMaxentSolver for solving +/-1 Ising model.

    Value:
    ------
    calc_e,mch_approximation
    """
    # Defime functions necessary for solving.
    @jit(nopython=True,cache=True)
    def fast_sum(J,s):
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
        Params:
        -------
        s (2D ndarray)
            state either {0,1} or {+/-1}
        params (ndarray)
            (h,J) vector
        """
        e = -fast_sum(params[s.shape[1]:],s)
        e -= np.dot(s,params[:s.shape[1]])
        return e

    @jit
    def mch_approximation( samples, dlamda ):
        dE = calc_e(samples,dlamda)
        dE -= dE.min()
        ZFraction = 1. / np.mean(np.exp(-dE))
        predsisj=pair_corr( samples, weighted=np.exp(-dE)/len(dE),concat=True ) * ZFraction  
        if np.any(predsisj<-1.00000001) or np.any(predsisj>1.000000001):
            print(predsisj.min(),predsisj.max())
            raise Exception("Predicted values are beyond limits.")
        return predsisj
    return calc_e,mch_approximation


