# 2014-01-25
import numpy as np
import numbapro as nb

def calc_e(J, samples):
    """
    2014-01-25
    """
    # if only one sample was given must get dimensions differently from vector
    if samples.ndim==1:
        N = samples.size
        S = 1
        samples = np.reshape(samples,(1,N))
    else:
        N = samples.shape[1]
        S = samples.shape[0]

    # initialize
    NN = N*(N-1)/2
    e = np.zeros((S))

    for s in range(S):
        k = 0
        for i in range(N):
            e[s] += J[i+NN]*samples[s,i]
            for j in range(i+1,N):
                e[s] += J[k]*samples[s,i]*samples[s,j]
                k+=1
    return -e

#def sample_metropolis(np.ndarray[dtype=np.float_t,ndim=2] JMat,
#                       np.ndarray[dtype=np.int_t,ndim=2] sample0):
#    """
#        Metropolis sampling with return of the de for faster calculating of energy of new
#        state.
#
#        sample0: must be a 2d array
#    2014-01-26
#    """
#    cdef int randix
#    cdef float de
#    # JMat = JMat-np.mean(JMat[np.triu_indices(sample0.size)])
#
#    randix = np.random.randint(sample0.shape[1])
#    sample0[0,randix] = (sample0[0,randix]-1)*-1
#
#    if sample0[0,randix]==0:
#        de = np.sum(JMat[randix,:]*sample0) +JMat[randix,randix]
#    else:
#        de = -np.sum(JMat[randix,:]*sample0)
#
#    # Only accept flip if dE<=0 or probability exp(-dE)
#    # Thus reject flip if dE>0 and with probability (1-exp(-dE))
#    if (de>0 and (np.random.rand()>np.exp(-de))):
#        sample0[0,randix] = (sample0[0,randix]-1)*-1
#        return 0.
#    else:
#        return de

@nb.jit('(nb.float64, nb.int32[:])(nb.float64[:,:], nb.int32[:], nb.int32, nb.float64)',
     target="gpu")
def sample_metropolis_wrand1(JMat, sample0, randix, r):
    """
        Metropolis sampling with return of the de for faster calculating of energy of new
        state and return sample.

        sample0: must be a 2d array
        2014-05-06
    """
    sample0[randix] = (sample0[randix]-1)*-1

    if sample0[randix]==0:
        de = np.sum(JMat[randix,:]*sample0) +JMat[randix,randix]
    else:
        de = -np.sum(JMat[randix,:]*sample0)

    # Only accept flip if dE<=0 or probability exp(-dE)
    # Thus reject flip if dE>0 and with probability (1-exp(-dE))
    if (de>0 and (r>np.exp(-de))):
        sample0[randix] = (sample0[randix]-1)*-1
        return 0.,sample0
    else:
        return de,sample0

#def sample_metropolis_wrand(np.ndarray[dtype=np.float_t,ndim=2] JMat,
#                      np.ndarray[dtype=np.int_t,ndim=2] sample0,
#                      int randix, float r):
#    """
#        Metropolis sampling with return of the de for faster calculating of energy of new
#        state.
#
#        sample0: must be a 2d array
#        2014-05-05
#    """
#    cdef float de
#    # JMat =JMat-np.mean(JMat[np.triu_indices(sample0.size)])
#
#    sample0[0,randix] = (sample0[0,randix]-1)*-1
#
#    if sample0[0,randix]==0:
#        de = np.sum(JMat[randix,:]*sample0) +JMat[randix,randix]
#    else:
#        de = -np.sum(JMat[randix,:]*sample0)
#
#    # Only accept flip if dE<=0 or probability exp(-dE)
#    # Thus reject flip if dE>0 and with probability (1-exp(-dE))
#    if (de>0 and (r>np.exp(-de))):
#        sample0[0,randix] = (sample0[0,randix]-1)*-1
#        return 0.
#    else:
#        return de
#
#def equilibrate_samples(np.ndarray[np.float_t,ndim=1] J,
#                        np.ndarray[np.int_t,ndim=2] sample0,
#                        int iters, int S,
#                        int burnin=200):
#    """
#        2014-01-26
#    """
#    cdef int N, NN, s, j
#    cdef float e0, de
#    cdef np.ndarray[np.float_t,ndim=1] e
#    cdef np.ndarray[np.float_t,ndim=2] JMat
#    cdef np.ndarray[np.int_t,ndim=2] samples
#
#    N = sample0.shape[1]
#    NN = N*(N-1)/2
#    e0 = calc_e(J,sample0)
#    samples = np.zeros((S,N),dtype=np.int64)
#    e = np.zeros((S))
#    JMat = convert_utri_to_array(J[:NN],J[NN:],N)
#    s = 0
#
#    # First, burn in.
#    for j in range(burnin):
#        de = sample_metropolis(JMat,sample0)
#        e0 += de
#
#    # Now, sample.
#    while s<S:
#        for j in range(iters):
#            de = sample_metropolis(JMat,sample0)
#            e0 += de
#        samples[s,:] = sample0.copy()
#        e[s] = e0
#        s += 1
#    return (e,samples)
#
#def equilibrate_samples_rand_states(np.ndarray[np.float_t,ndim=1] J,
#                                    int N, int iters, int S,
#                                    int burnin=200):
#    """
#        Sample state space starting at a random point every time. This is a more efficient
#        sampling of the space when we have a large spin system.
#    2014-01-27
#    """
#    cdef int NN, s, j, k
#    cdef float e0, de
#    cdef np.ndarray[np.float_t,ndim=1] e
#    cdef np.ndarray[np.float_t,ndim=2] JMat
#    cdef np.ndarray[np.int_t,ndim=2] samples
#
#    NN = N*(N-1)/2
#    samples = np.random.randint(2,size=(S,N))
#    e = np.zeros((S))
#    JMat = convert_utri_to_array(J[:NN],J[NN:],N)
#    s,k = 0, 0
#
#    # Generate random numbers in vectorized form to speed computations.
#    randix = np.random.randint(N,size=(S*iters+S*burnin))
#    r = np.random.rand(S*iters+S*burnin)
#    
#    # Sample.
#    while s<S:
#        # First, burn in.
#        for j in range(burnin):
#            sample_metropolis_wrand( JMat,np.reshape(samples[s,:],(1,N)),
#                                     randix[k], r[k] );
#            k += 1
#        e0 = calc_e( J,np.reshape(samples[s,:],(1,N)) )
#
#        # Metropolis sample.
#        for j in range(iters):
#            de = sample_metropolis_wrand( JMat,np.reshape(samples[s,:],(1,N)),
#                                          randix[k], r[k] )
#            e0 += de
#            k += 1
#        e[s] = e0
#        s += 1
#    return (e,samples)
#
#def sample_f_save_e(int s, np.ndarray[np.int_t,ndim=1] sample, np.ndarray[np.float_t,ndim=2] JMat,
#             np.ndarray[np.float_t,ndim=1] J, int iters ):
#    """
#        Sample using Metropolis and return energy of every state through sampling.
#    2014-05-06
#    """
#    cdef np.ndarray[np.int_t,ndim=1] randix
#    cdef np.ndarray[np.float_t,ndim=1] r
#    cdef np.ndarray[np.float_t,ndim=1] e0
#    cdef int j,k,N,k0
#    cdef float de
#    N = sample.size
#
#    # Generate random numbers in vectorized form to speed computations.
#    randix = np.random.randint(N,size=(iters))
#    r = np.random.rand(iters)
#    e0 = np.zeros((iters+1))
#    k = 0
#    
#    # First, burn in.
##    for j in range(burnin):
##        sample = sample_metropolis_wrand1( JMat,sample,
##                                           randix[k], r[k] )[1]
##        k += 1
#    k0 = k
#    e0[k-k0] = calc_e( J,np.reshape(sample,(1,N)) )
#
#    # Metropolis sample.
#    for j in range(iters):
#        de,sample = sample_metropolis_wrand1( JMat,sample,
#                                      randix[k], r[k] )
#        e0[k+1-k0] = e0[k-k0] + de
#        k += 1
#    return e0,sample
#
#def sample_f(int s, np.ndarray[np.int_t,ndim=1] sample, np.ndarray[np.float_t,ndim=2] JMat,
#             np.ndarray[np.float_t,ndim=1] J, int iters, int burnin ):
#    """
#    2014-05-06
#    """
#    cdef np.ndarray[np.int_t,ndim=1] randix
#    cdef np.ndarray[np.float_t,ndim=1] r
#    cdef int j,k,N
#    cdef float e0,de
#    N = sample.size
#
#    # Generate random numbers in vectorized form to speed computations.
#    randix = np.random.randint(N,size=(iters+burnin))
#    r = np.random.rand(iters+burnin)
#    k = 0
#    
#    # First, burn in.
#    for j in range(burnin):
#        sample = sample_metropolis_wrand1( JMat,sample,
#                                 randix[k], r[k] )[1]
#        k += 1
#    e0 = calc_e( J,np.reshape(sample,(1,N)) )
#
#    # Metropolis sample.
#    for j in range(iters):
#        de,sample = sample_metropolis_wrand1( JMat,sample,
#                                      randix[k], r[k] )
#        e0 += de
#        k += 1
#    return e0,sample
#
#def equilibrate_samples_save_e(np.ndarray[dtype=np.float_t,ndim=1] J,
#                               np.ndarray[dtype=np.int_t,ndim=2] samples,
#                               int iters):
#    """
#        Equilibrate using Metropolis, but save and return energy of each state.
#    2014-01-25
#    """
#    cdef int S, N, NN, s, j
#    cdef float de, e0
#    cdef np.ndarray[dtype=np.float_t,ndim=2] save_e, JMat
#
#    S,N = samples.shape[0], samples.shape[1]
#    NN = N*(N-1)/2
#    save_e = np.zeros((S,iters))
#    JMat = convert_utri_to_array(J[:NN],J[NN:],N)
#    e0 = 0.
#
#    for s in range(S):
#        e0 = calc_e(J,np.reshape(samples[s,:],(1,N)))
#        save_e[s,0] = e0
#        for j in range(1,iters):
#            save_e[s,j] = save_e[s,j-1] +\
#                                sample_metropolis(JMat,np.reshape(samples[s,:],(1,N)))
#    return (save_e,samples)
#
#def convert_utri_to_array(np.ndarray[np.float_t,ndim=1] vec,
#                          np.ndarray[np.float_t,ndim=1] diag, int N):
#    """
#        Take a vector of the upper triangle and convert it to an array with
#        diagonal elements given separately.
#    2014-01-25
#    """
#    cdef np.ndarray[np.float_t,ndim=2] mat
#    cdef int k, i, j
#
#    mat = np.zeros((N,N))
#    k = 0
#    for i in range(N-1):
#        for j in range(i+1,N):
#            mat[i,j] = vec[k]
#            k+=1
#
#    mat = mat+np.transpose(mat)
#    mat[np.eye(N)==1] = diag
#    return mat
#
#def hello():
#    print "Hello"
#    return 1.
#
