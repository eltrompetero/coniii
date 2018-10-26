# =============================================================================================== #
# meanFieldIsing.py
# Author : Bryan Daniels
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
# =============================================================================================== #
import scipy
import scipy.optimize
import copy
#import scipy.weave # for efficient fourth-order matrix calculation

exp, cosh = scipy.exp, scipy.cosh
dot = scipy.dot

# 4.8.2011
# 8.16.2012 moved from generateFightData.py
def aboveDiagFlat(mat, keepDiag=False, offDiagMult=None):
    """
    Return a flattened list of all elements of the 
    matrix above the diagonal.
    
    Use offDiagMult = 2 for symmetric J matrix.
    """

    m = copy.copy(mat)
    if offDiagMult is not None:
        m *= offDiagMult*(1.-scipy.tri(len(m)))+scipy.diag(scipy.ones(len(m))) 
    if keepDiag: begin=0
    else: begin=1
    return scipy.concatenate([ scipy.diagonal(m,i)                          \
                              for i in range(begin,len(m)) ])

# 9.15.2014 updated for new scipy.diag behavior
# 1.17.2013 moved from criticalPoint.py
# 1.31.2012
def replaceDiag(mat,lst):
    if len(scipy.shape(lst)) > 1:
        raise Exception("Lst should be 1-dimensional")
    if scipy.shape(mat) != (len(lst),len(lst)):
        raise Exception("Incorrect dimensions."+                   \
            "  shape(mat) = "+str(scipy.shape(mat))+                \
            ", len(lst) = "+str(len(lst)))
    return mat - scipy.diag(scipy.diag(mat).copy()).copy()          \
        + scipy.diag(lst).copy()

# 2.15.2013 moved from branchingProcess.py
# 2.11.2013
def zeroDiag(mat):
    return replaceDiag(mat,scipy.zeros(len(mat)))


def m(h,J,ell,T):
    """
    Careful if T is small for loss of precision?
    """
    func = lambda mm: mm - 1./(1.+exp((h+2.*(ell-1.)*mm*J)/T))
    #dfunc = lambda m: 1. - (ell-1.)*J/T /                   \
    #    ( cosh((h-(ell-1.)*m*J)/(2.*T)) )**2
    #***
    #ms = scipy.linspace(-0.1,1.1,100)
    #pylab.plot(ms,[func(m) for m in ms])
    #***
    #mRoot0 = 0.5
    
    mRoot = scipy.optimize.brentq(func,-0.,1.)
    return mRoot
    
def avgE(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    
    return -mr*hloc
    
def dmdT(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    
    return hloc/(4.*T**2) / cosh(-hloc/(2.*T))**2 / (1.+(ell-1)*J/(4.*T)*cosh(-hloc/(2.*T))**-2)
    
def specificHeat(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    denomFactor = 4.*T/(ell-1)/J * cosh(-hloc/(2.*T))**2
    
    return hloc*(2*(ell-1)*mr*J - h)
    """
      /                      \
        ( T*(ell-1)*J*                                      \
             ( 1. + denomFactor ) * \
             ( 1. + 1./denomFactor ) )
    """

def susc(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h

    return 1./(T**2 * (exp(-hloc) + exp(+hloc) + 2.))

def coocCluster(coocMat, cluster):
    """Sort coocMat by the cluster indices"""
    orderedIndices = cluster
    sortedMat = scipy.array(coocMat)[:]
    sortedMat = sortedMat[orderedIndices,:]
    sortedMat = sortedMat[:,orderedIndices]
    return sortedMat

# 3.27.2014 moved from selectiveClusterExpansion.py
def JfullFromCluster(Jcluster,cluster,N):
    """
    NOTE: There is perhaps a faster way of doing this?
    """
    J = scipy.zeros((N,N))
    for i,iFull in enumerate(cluster):
        for j,jFull in enumerate(cluster):
            J[iFull,jFull] = Jcluster[i,j]
    return J

# 3.27.2014 moved from selectiveClusterExpansion.py
def symmetrizeUsingUpper(mat):
    if len(mat) != len(mat[0]): raise Exception
    d = scipy.diag(mat)
    matTri = (1.-scipy.tri(len(mat)))*mat
    matSym = replaceDiag(matTri+matTri.T,d)
    return matSym

# 3.27.2014 moved from selectiveClusterExpansion.py
# Eqs. 30-33
def SmeanField(cluster,coocMat,meanFieldPriorLmbda=0.,
    numSamples=None,indTerm=True,alternateEnt=False,
    useRegularizedEq=True):
    """
    meanFieldPriorLmbda (0.): 3.23.2014
    indTerm (True)          : As of 2.19.2014, I'm not
                              sure whether this term should
                              be included, but I think so
    alternateEnt (False)    : Explicitly calculate entropy
                              using the full partition function
    useRegularizedEq (True) : Use regularized form of equation
                              even when meanFieldPriorLmbda = 0.
    """
    
    coocMatCluster = coocCluster(coocMat,cluster)
    # in case we're given an upper-triangular coocMat:
    coocMatCluster = symmetrizeUsingUpper(coocMatCluster)
    
    outer = scipy.outer
    N = len(cluster)
    
    freqs = scipy.diag(coocMatCluster)
    c = coocMatCluster - outer(freqs,freqs)
    
    Mdenom = scipy.sqrt( outer(freqs*(1.-freqs),freqs*(1-freqs)) )
    M = c / Mdenom
    
    if indTerm:
        Sinds = -freqs*scipy.log(freqs)             \
            -(1.-freqs)*scipy.log(1.-freqs)
        Sind = scipy.sum(Sinds)
    else:
        Sind = 0.
    
    # calculate off-diagonal (J) parameters
    if (meanFieldPriorLmbda != 0.) or useRegularizedEq:
        # 3.22.2014
        if meanFieldPriorLmbda != 0.:
            gamma = meanFieldPriorLmbda / numSamples
        else:
            gamma = 0.
        mq,vq = scipy.linalg.eig(M)
        mqhat = 0.5*( mq-gamma +                        \
                scipy.sqrt((mq-gamma)**2 + 4.*gamma) )
        jq = 1./mqhat #1. - 1./mqhat
        Jprime = scipy.real_if_close(                   \
                dot( vq , dot(scipy.diag(jq),vq.T) ) )
        JMF = zeroDiag( Jprime / Mdenom )
        
        ent = scipy.real_if_close(                      \
                Sind + 0.5*scipy.sum( scipy.log(mqhat)  \
                + 1. - mqhat ) )
    else:
        # use non-regularized equations
        Minv = scipy.linalg.inv(M)
        JMF = zeroDiag( Minv/Mdenom )
        
        logMvals = scipy.log( scipy.linalg.svdvals(M) )
        ent = Sind + 0.5*scipy.sum(logMvals)
    
    # calculate diagonal (h) parameters
    piFactor = scipy.repeat( [(freqs-0.5)/(freqs*(1.-freqs))],
                            N, axis=0).T
    pjFactor = scipy.repeat( [freqs], N, axis=0 )
    factor2 = c*piFactor - pjFactor
    hMF = scipy.diag( scipy.dot( JMF, factor2.T  ) ).copy()
    if indTerm:
        hMF -= scipy.log(freqs/(1.-freqs))
    
    J = replaceDiag( 0.5*JMF, hMF )
    
    if alternateEnt:
        ent = analyticEntropy(J)
    
    # make 'full' version of J (of size NfullxNfull)
    Nfull = len(coocMat)
    Jfull = JfullFromCluster(J,cluster,Nfull)
    
    return ent,Jfull


# 11.20.2014 for convenience
def JmeanField(coocMat,**kwargs):
    """
    See SmeanField for important optional arguments,
    including noninteracting prior weighting.
    """
    ell = len(coocMat)
    S,JMF = SmeanField(list(range(ell)),coocMat,**kwargs)
    return JMF


# 11.21.2014
def meanFieldStability(J,freqs):
    # 6.26.2013
    #freqs = scipy.mean(samples,axis=0)
    f = scipy.repeat([freqs],len(freqs),axis=0)
    m = -2.*zeroDiag(J)*f*(1.-f)
    stabilityValue = max(abs( scipy.linalg.eigvals(m) ))
    return stabilityValue


# 3.6.2015
# exact form of log(cosh(x)) that doesn't die at large x
def logCosh(x):
    return abs(x) + scipy.log(1. + scipy.exp(-2.*abs(x))) - scipy.log(2.)

# 3.6.2015
# see notes 3.4.2015
def FHomogeneous(h,J,N,m):
    """
    Use Hubbard-Stratonovich (auxiliary field) to calculate the
    (free energy?) of a homogeneous system as a function of the
    field m (m equals the mean field as N -> infinity?).
    """
    Jbar = N*J
    s = scipy.sqrt(scipy.pi/(N*Jbar))
    L = Jbar * m*m - scipy.log(2.) - logCosh(2.*Jbar*m + h)
    return N*L + scipy.log(s)

# 3.6.2015
def dFdT(h,J,N,m):
    Jbar = N*J
    return -N*Jbar*m*m + N*(2.*Jbar*m + h)*scipy.tanh(2.*Jbar*m + h) + 0.5

# 3.6.2015
def SHomogeneous(h,J,N):
    """
    Use Hubbard-Stratonovich (auxiliary field) to numerically 
    calculate entropy of a homogeneous system.
    """
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]

    dFdTFunc = lambda m: dFdT(h,J,N,m) * scipy.exp(-FHomogeneous(h,J,N,m))
    avgdFdT = scipy.integrate.quad(dFdTFunc,-scipy.inf,scipy.inf)[0] / Z

    return scipy.log(Z) - avgdFdT

# 3.6.2015
def avgmHomogeneous(h,J,N):
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]
    
    mFunc = lambda m: m * scipy.exp(-FHomogeneous(h,J,N,m))
    avgm = scipy.integrate.quad(mFunc,-scipy.inf,scipy.inf)[0] / Z

    return avgm

# 3.6.2015
def avgxHomogeneous(h,J,N):
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]
    
    Jbar = N*J
    xFunc = lambda m: scipy.tanh(2.*Jbar*m+h) * scipy.exp(-FHomogeneous(h,J,N,m))
    avgx = scipy.integrate.quad(xFunc,-scipy.inf,scipy.inf)[0] / Z
    
    return avgx

# 3.6.2015
def multiInfoHomogeneous(h,J,N):
    Sind = independentEntropyHomogeneous(h,J,N)
    S = SHomogeneous(h,J,N)
    return Sind - S

# 3.6.2015
def independentEntropyHomogeneous(h,J,N):
    avgx = avgxHomogeneous(h,J,N)
    S1 = - (1.+avgx)/2. * scipy.log((1.+avgx)/2.) \
        - (1.-avgx)/2. * scipy.log((1.-avgx)/2.)
    return N*S1

# 3.6.2015
def independentEntropyHomogeneous2(h,J,N):
    avgx = avgxHomogeneous(h,J,N)
    heff = scipy.arctanh(avgx)
    return N*(scipy.log(2.) + scipy.log(scipy.cosh(heff)) - avgx*heff)


# 7.18.2017 moved from inverseIsing.py
# 7.6.2012
def findJmatrixAnalytic_CoocMat(coocMatData,
                                Jinit=None,
                                bayesianMean=False,
                                numSamples=None,
                                priorLmbda=0.,
                                minSize=0):
    
    ell = len(coocMatData)
    
    if priorLmbda != 0.:
        lmbda = priorLmbda / numSamples
    else:
        lmbda = 0.
    
    if bayesianMean:
        coocMatDesired = coocMatBayesianMean(coocMatData,numSamples)
    else:
        coocMatDesired = coocMatData
    
    if Jinit is None:
        # 1.17.2012 try starting from frequency model
        freqs = scipy.diag(coocMatDesired)
        hList = -scipy.log(freqs/(1.-freqs))
        Jinit = scipy.diag(hList)
    
    def deltaCooc(Jflat):
        J = unflatten(Jflat,ell)
        cooc = coocExpectations(J,minSize=minSize)
        dCooc = aboveDiagFlat(cooc - coocMatDesired,keepDiag=True)
        if (lmbda > 0.) and (ell > 1):
            freqs = scipy.diag(coocMatDesired)
            factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
            factorFlat = aboveDiagFlat(factor)
            # 3.24.2014 changed from lmbda/2. to lmbda
            priorTerm = lmbda * factorFlat * Jflat[ell:]**2 
            dCooc = scipy.concatenate([dCooc,priorTerm])
        return dCooc
    
    JinitFlat = aboveDiagFlat(Jinit,keepDiag=True)
    Jflat = scipy.optimize.leastsq(deltaCooc,JinitFlat)[0]
    Jnew = unflatten(Jflat,ell,symmetrize=True)
    
    return Jnew

# 7.18.2017 moved from inverseIsing.py
def unflatten(flatList, ell, symmetrize=False):
    """
    Inverse of aboveDiagFlat with keepDiag=True.
    """
    
    mat = scipy.sum([ scipy.diag(flatList[diagFlatIndex(0,j,ell):diagFlatIndex(0,j+1,ell)], k=j)
                      for j in range(ell)], axis=0)
    if symmetrize:
        return 0.5*(mat + mat.T)
    else:
        return mat

# 7.18.2017 moved from inverseIsing.py
def diagFlatIndex(i,j,ell):
    """
    Should have j>=i...
    """
    D = j-i
    return i + D*ell - D*(D-1)//2

# 7.18.2017 moved from inverseIsing.py
# 2.18.2014
def analyticEntropy(J):
    """
    In nats.
    """
    Z = unsummedZ(J)
    p = Z / scipy.sum(Z)
    return - scipy.sum( p * scipy.log(p) )

# 7.20.2017 moved from inverseIsing.py
# 2.7.2014
def coocSampleCovariance(samples,bayesianMean=True,includePrior=True):
    """
    includePrior (True)             : Include diagonal component corresponding
                                      to ell*(ell-1)/2 prior residuals for
                                      interaction parameters
    """
    coocs4 = fourthOrderCoocMat(samples)
    if bayesianMean:
        #coocs4mean = coocMatBayesianMean(coocs4,len(samples))
        print("coocSampleCovariance : WARNING : using ad-hoc 'Laplace' correction")
        N = len(samples)
        newDiag = (scipy.diag(coocs4)*N + 1.)/(N + 2.)
        coocs4mean = replaceDiag(coocs4,newDiag)
    else:
        coocs4mean = coocs4
    cov = coocs4mean*(1.-coocs4mean)
    if includePrior:
        ell = len(samples[0])
        one = scipy.ones(ell*(ell-1)//2)
        return scipy.linalg.block_diag( cov, scipy.diag(one) )
    else:
        return cov

# 7.20.2017 moved from inverseIsing.py
# 4.11.2011
# 1.12.2012 changed to take coocMatDesired instead of dataSamples
def isingDeltaCooc(isingSamples,coocMatDesired):
    isingCooc = cooccurranceMatrixFights(isingSamples,keepDiag=True)
    #dataCooc = cooccurranceMatrixFights(dataSamples,keepDiag=True)
    return aboveDiagFlat(isingCooc-coocMatDesired,keepDiag=True)

# 7.18.2017 from SparsenessTools.cooccurranceMatrixFights
# 3.17.2014 copied from generateFightData.py
# 4.1.2011
def cooccurrence_matrix(samples,keepDiag=True):
    """
    """
    samples = scipy.array(samples,dtype=float)
    mat = scipy.dot(samples.T,samples)
    if keepDiag: k=-1
    else: k=0
    mat *= (1 - scipy.tri(len(mat),k=k)) # only above diagonal
    mat /= float(len(samples)) # mat /= np.sum(mat)
    return mat

# 7.20.2017 moved from inverseIsing.py
# 7.20.2017 TO DO: change to slowMethod=False by incorporating weave or something else
# 2.17.2012
def fourthOrderCoocMat(samples, slowMethod=True):
    ell = len(samples[0])
    samples = scipy.array(samples)
    jdim = (ell+1)*ell//2
    f = scipy.zeros((jdim, jdim))
    
    if slowMethod:
        for i in range(ell):
          for j in range(i,ell):
            for m in range(i,ell):
              for n in range(m,ell):
                coocIndex1 = diagFlatIndex(i,j,ell)
                coocIndex2 = diagFlatIndex(m,n,ell)
                cooc = scipy.sum(                                           \
                    samples[:,i]*samples[:,j]*samples[:,m]*samples[:,n])
                f[coocIndex1,coocIndex2] = cooc
                f[coocIndex2,coocIndex1] = cooc
    else:
        code = """
        int coocIndex1,coocIndex2;
        float coocSum;
        for (int i=0; i<ell; i++){
          for (int j=i; j<ell; j++){
            for (int m=i; m<ell; m++){
              for (int n=m; n<ell; n++){
                coocIndex1 = i + (j-i)*ell - (j-i)*(j-i-1)/2;
                coocIndex2 = m + (n-m)*ell - (n-m)*(n-m-1)/2;
                coocSum = 0.;
                for (int k=0; k<numFights; k++){
                  coocSum += samples(k,i)*samples(k,j)*samples(k,m)*samples(k,n);
                }
                f(coocIndex1,coocIndex2) = coocSum;
                f(coocIndex2,coocIndex1) = coocSum;
              }
            }
          }
        }
        """
        numFights = len(samples)
        err = scipy.weave.inline(code,                                      \
            ['f','samples','numFights','ell'],                              \
            type_converters = scipy.weave.converters.blitz)
    return f/float(len(samples))

# 7.20.2017 moved from inverseIsing.py
# 1.30.2012
def seedGenerator(seedStart,deltaSeed):
    while True:
        seedStart += deltaSeed
        yield seedStart

# 7.25.2017 moved from inverseIsing.py
# 2.21.2013
def coocStdevsFlat(coocMat,numFights):
    """
    Returns a flattened expected standard deviation matrix used
    to divide deltaCooc to turn it into z scores.
    """
    coocMatMean = coocMatBayesianMean(coocMat,numFights)
    varianceMatFlat = aboveDiagFlat(coocMatMean*(1.-coocMatMean)/numFights,keepDiag=True)
    return scipy.sqrt(varianceMatFlat)

# 7.25.2017 moved from inverseIsing.py
# 3.5.2013
def coocMatBayesianMean(coocMat,numFights):
    """
    Using "Laplace's method"
    """
    return (coocMat*numFights + 1.)/(numFights + 2.)

# 7.25.2017 moved from inverseIsing.py
# 4.11.2011
# 1.12.2012 changed to take coocMatDesired instead of dataSamples
def isingDeltaCooc(isingSamples,coocMatDesired):
    isingCooc = cooccurrence_matrix(isingSamples)
    return aboveDiagFlat(isingCooc-coocMatDesired,keepDiag=True)


# --- exact Ising code below ---
# 7.18.2017 For now, this uses Bryan's code.  Could be updated to use coniii's
# exact inverse ising solver.

# 7.18.2017 moved from inverseIsing.py
def coocExpectations(J,hext=0,zeroBelowDiag=True,minSize=0):
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    coocp = scipy.array([ scipy.outer(f,f) for f in fp ])
    Z = unsummedZ(J,hext,minSize)
    coocSym = dot(coocp.T,Z)/sum(Z)
    if zeroBelowDiag:
        coocTri = coocSym * scipy.tri(ell).T
        return coocTri
    else:
        return coocSym

# 7.18.2017 moved from inverseIsing.py
def unsummedZ(J,hext=0,minSize=0):
    """
    J should have h on the diagonal.
    """
    return scipy.exp( unsummedLogZ(J,hext=hext,minSize=minSize) )

# 7.18.2017 moved from inverseIsing.py
def unsummedLogZ(J,hext=0,minSize=0):
    """
    J should have h on the diagonal.
    """
    ell = len(J)
    h = scipy.diag(J)
    JnoDiag = J - scipy.diag(h)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    return -dot(fp,h-hext)-1.0*scipy.sum(dot(fp,JnoDiag)*fp,axis=1)

# 7.18.2017 moved from inverseIsing.py
def fightPossibilities(ell,minSize=0):
    fightNumbers = list(range(2**ell))
    fp = [ [ int(x) for x in scipy.binary_repr(fN,ell) ]                  \
             for fN in fightNumbers ]
    if minSize > 0:
        fp = scipy.array( [x for x in fp if sum(x)>=minSize] )
    return fp
