"""Pseudolikelihood inverse-Ising prototype — LEGACY.

A stand-alone implementation of the pseudolikelihood approach to the
inverse Ising problem (Aurell and Ekeberg, PRL 108, 090201 (2012)).
Not wrapped by any solver class in :mod:`coniii.solvers` and not
exported from the top-level :mod:`coniii`. Kept for reference;
scheduled for removal in coniii v5.

Direct imports emit a :class:`DeprecationWarning`.

Originally authored by Bryan C. Daniels.
"""
import warnings as _warnings

_warnings.warn(
    "coniii.legacy.pseudo_inverse_ising is legacy code that no "
    "solver class wraps. It may be removed in coniii v5. For "
    "pseudo-likelihood inverse Ising, use coniii.solvers.Pseudo.",
    DeprecationWarning,
    stacklevel=2,
)

#import scipy.weave  # retired in scipy 1.0
#import sys
import scipy.optimize
import numpy as np

import pylab # for testing

exp,log,sum,array = np.exp,np.log,np.sum,np.array

def pseudoInverseIsing(samples,minSize=0):
    """
    
    minSize (0) : minimum number of participants per sample
                  (set to 2 for fights)
    """

    data = array( [f for f in samples if sum(f) >= minSize] )
    ell = len(data[0])
    
    # start at freq. model params?
    freqs = np.mean(data,axis=0)
    hList = -np.log(freqs/(1.-freqs))

    Jfinal = np.zeros((ell,ell))

    for r in range(ell):
        
        print("Minimizing for r =",r)
        
        Jr0 = np.zeros(ell) #np.ones(ell)
        Jr0[r] = hList[r]
        
        # 12.10.2013
        samplesRhat = data.copy()
        samplesRhat[:,r] = np.ones(len(data))
        # calculate once and pass to hessian algorithm for speed
        pairCoocRhat = pairCoocMat(samplesRhat)
        
        Lr = lambda Jr:                                             \
            - conditionalLogLikelihood(r,data,Jr,minSize=minSize)
        fprime = lambda Jr:                                         \
            conditionalJacobian(r,data,Jr,minSize=minSize)
        fhess = lambda Jr:                                          \
            conditionalHessian(r,data,Jr,minSize=minSize,           \
                                         pairCoocRhat=pairCoocRhat)
        
        Jr = scipy.optimize.fmin_ncg(Lr,Jr0,fprime,fhess=fhess)
    
        #return Jr

        Jfinal[r] = Jr

    Jfinal = 0.5*( Jfinal + Jfinal.T )

    return Jfinal



def conditionalLogLikelihood(r,samples,Jr,minSize=0):
    """
    (Equals -L_r from my notes.)
    
    r           : individual index
    samples     : binary matrix, (# samples) x (dimension of system)
    Jr          : (dimension of system) x (1)
    minSize (0) : minimum number of participants (set to 2 for fights)
    """
    samples,Jr = array(samples),array(Jr)
    
    sigmaRtilde = (2.*samples[:,r] - 1.)
    samplesRhat = 2.*samples.copy()
    samplesRhat[:,r] = np.ones(len(samples))
    localFields = np.dot(Jr,samplesRhat.T) # (# samples)x(1)
    energies = sigmaRtilde * localFields # (# samples)x(1)
    
    # vector with zeros on samples affected by minSize
    filterVec = 1 - samples[:,r] * ( sum(samples,axis=1) <= minSize )

    invPs = 1. + exp( energies )
    logLs = - filterVec * log( invPs )

    return np.sum( logLs )

def conditionalJacobian(r,samples,Jr,minSize=0):
    """
    Returns d conditionalLogLikelihood / d Jr,
    with shape (dimension of system)
    """
    samples,Jr = array(samples),array(Jr)
    ell = len(Jr)
    
    sigmaRtilde = (2.*samples[:,r] - 1.)
    samplesRhat = 2.*samples.copy()
    samplesRhat[:,r] = np.ones(len(samples))
    localFields = np.dot(Jr,samplesRhat.T) # (# samples)x(1)
    energies = sigmaRtilde * localFields # (# samples)x(1)
    
    # vector with zeros on samples affected by minSize
    filterVec = 1 - samples[:,r] * ( sum(samples,axis=1) <= minSize )
    
    coocs = np.repeat([sigmaRtilde],ell,axis=0).T * samplesRhat # (#samples)x(ell)

    return np.dot( coocs.T, filterVec * 1./(1. + exp(-energies)) )

def conditionalHessian(r,samples,Jr,minSize=0,pairCoocRhat=None):
    """
    Returns d^2 conditionalLogLikelihood / d Jri d Jrj,
    with shape (dimension of system)x(dimension of system)
    
    pairCooc (None)     : Pass pairCoocMat(samples) to speed
                          calculation.
    
    Current implementation uses more memory for speed.
    For large #samples, it may make sense to break up differently
    if too much memory is being used.
    """
    samples,Jr = array(samples),array(Jr)
    ell = len(Jr)
    
    sigmaRtilde = (2.*samples[:,r] - 1.)
    samplesRhat = 2.*samples.copy()
    samplesRhat[:,r] = np.ones(len(samples))
    localFields = np.dot(Jr,samplesRhat.T) # (# samples)x(1)
    energies = sigmaRtilde * localFields # (# samples)x(1)
    
    # pairCooc has shape (# samples)x(ell)x(ell)
    if pairCoocRhat is None:
        pairCoocRhat = pairCoocMat(samplesRhat)
    
    # vector with zeros on samples affected by minSize
    filterVec = 1 - samples[:,r] * ( sum(samples,axis=1) <= minSize )

    energyMults = exp(-energies)/( (1.+exp(-energies))**2 ) # (# samples)x(1)
    #filteredSigmaRtildeSq = filterVec * (2.*samples[:,r] + 1.) # (# samples)x(1)
    filteredSigmaRtildeSq = filterVec # (sigmaRtildeSq = 1)

    #return energyMults,filteredSigmaR
    return np.dot( filteredSigmaRtildeSq * energyMults, pairCoocRhat )

def testDerivatives(r,i,samples,J,minSize=0,deltaMax=1):
    Jr0 = J[r]
    ell = len(Jr0)
    
    # set up perturbations
    v = np.zeros(ell)
    v[i] = 1
    deltas = np.linspace(-deltaMax,deltaMax,101)
    Jrs = [ Jr0 + v*delta for delta in deltas ]

    # calculate numerically
    negLls = [ -conditionalLogLikelihood(r,samples,Jr,minSize=minSize) \
              for Jr in Jrs ]

    # calculate analytically
    jac = conditionalJacobian(r,samples,Jr0,minSize=minSize)
    hess = conditionalHessian(r,samples,Jr0,minSize=minSize)

    pylab.figure()
    pylab.plot(deltas,negLls,'o:')
    pylab.plot(deltas,[negLls[50] + jac[i]*delta \
                       for delta in deltas],'g-')
    pylab.plot(deltas,[negLls[50] + jac[i]*delta + 0.5*hess[i,i]*delta**2 \
                       for delta in deltas],'r-')

    #pylab.figure()
    #pylab.plot(deltas,[negLls - jac[i]*delta for delta in deltas],'o:')
    #pylab.plot(deltas,[negLls[50] + 0.5*hess[i,i]*delta**2 \
    #                   for delta in deltas],'r-')

def pairCoocMat(samples):
    """
    Returns matrix of shape (ell)x(# samples)x(ell).
    
    For use with conditionalHessian.
    
    Slow because I haven't thought of a better way of doing it yet.
    """
    p = [ np.outer(f,f) for f in samples ]
    return np.transpose(p,(1,0,2))


def pseudoLogLikelihood(samples,J,minSize=0):
    """
    samples     : binary matrix, (# samples) x (dimension of system)
    J           : (dimension of system) x (dimension of system)
                : J should be symmetric
                
    (Could probably be made more efficient.)
    """
    return np.sum([ conditionalLogLikelihood(r,samples,J,minSize) \
                       for r in range(len(J)) ])



