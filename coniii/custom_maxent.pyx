# 2015-08-14
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from data_sets.us_congress.session import Session
import entropy.entropy as entropy

#session = Session(111,'s')
#v = session.dem_votes()
#n = v.shape[1]
#
#pMaj = np.bincount(np.sum( session.convert_to_maj(v)==-1, 1 ))
#pMaj = pMaj / np.sum(pMaj)
#constraints = np.concatenate(( np.mean(v,0),pMaj ))

cdef int n = 2

def calc_e( np.ndarray[np.float_t,ndim=1] s, np.ndarray[np.float_t,ndim=1] lamda):
    """Given state, compute energy."""
    cdef int i
    cdef np.ndarray[dtype=np.float_t,ndim=1] h = np.zeros((n))
    cdef np.ndarray[dtype=np.float_t,ndim=1] V = np.zeros((len(lamda)-n))
    
    h,V = lamda[:n],lamda[n:]
    return -( np.sum(s*h) + V[min([np.sum(s==-1),np.sum(s==1)])] )

def calc_observables( np.ndarray[np.float_t,ndim=2] samples):
    """Given data samples, calculate observables."""
    cdef np.ndarray[np.float_t,ndim=1] pMaj

    pMaj = np.bincount(np.min(np.hstack([np.sum(samples==1,1)[:,None],np.sum(samples==-1,1)[:,None]]),1),
                    minlength=np.ceil(n/2))*1.
    pMaj = pMaj / np.sum(pMaj)
    return np.concatenate(( np.mean(samples,0),pMaj ))

def mch_approximation( np.ndarray[np.float_t,ndim=2] samples, np.ndarray[np.float_t,ndim=1] dlamda ):
    cdef np.ndarray[np.float_t,ndim=1] s,dE,constraintsApproximation
    cdef float ZFraction 
    cdef np.ndarray[np.uint8_t,cast=True,ndim=1] ix
    cdef np.ndarray[np.int_t,ndim=1] nVotesInMaj
    cdef int i

    dE = np.array([calc_e(s,dlamda) for s in samples])
    dE -= np.min(dE)
    ZFraction = 1. / np.mean(np.exp(-dE))
    constraintsApproximation = np.zeros((len(dlamda)))
    
    constraintsApproximation[:n] = np.mean(samples*np.exp(-dE)[:,None],0) * ZFraction
    
    nVotesInMaj = np.min(np.hstack([np.sum(samples==1,1)[:,None],np.sum(samples==-1,1)[:,None]]),1)
    for i in xrange(int( np.ceil((n-1)/2) )+1):
        # Count only states with i votes in the majority.
        ix = nVotesInMaj==i
        constraintsApproximation[i+n] = np.sum(np.exp(-dE[ix]))/dE.size * ZFraction
    
    return constraintsApproximation

