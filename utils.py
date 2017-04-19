from __future__ import division
import numpy as np

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



# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_ising_mch_helpers():
    """
    Functions for plugging into GeneralMaxentSolver for solving +/-1 Ising model.
    2017-01-22

    Value:
    ------
    calc_e,mch_approximation
    """
    from mc_hist import GeneralMaxentSolver
    from entropy.entropy import calc_sisj

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
        predsisj=calc_sisj( samples, weighted=np.exp(-dE)/len(dE),concat=True ) * ZFraction  
        if np.any(predsisj<-1.00000001) or np.any(predsisj>1.000000001):
            print(predsisj.min(),predsisj.max())
            raise Exception("Predicted values are beyond limits.")
        return predsisj
    return calc_e,mch_approximation


