# 2017-09-30
from __future__ import division
from solvers import *
from utils import *
from samplers import MCIsing
import ising_eqn_5_sym
import importlib

# Define common functions.
calc_e,calc_observables,mch_approximation = define_ising_helpers_functions()

# Generate example data set.
#n = 5  # system size
#np.random.seed(0)
#h,J = np.random.normal(scale=.1,size=n),np.random.normal(scale=.1,size=n*(n-1)//2)  # random fields,couplings
#hJ = np.concatenate((h,J))
#p = ising_eqn_5_sym.p(hJ)  # probability distribution of all states p(s)
#sisjTrue = ising_eqn_5_sym.calc_observables(hJ)  # exact magnetizations and pairwise correlations

class Tester(object):
    def __init__(self,sample,lamda):
        """
        Classs using an analytic estimate of the gradient of the error to find the solution to the
        problem <<f_k>>-<f_k>=0.

        Parameters
        ----------
        sample : ndarray
            Sample of data.
        lamda : ndarray
            parameters
        """
        self.n = sample.shape[1]
        self.ising_eqns = importlib.import_module('ising_eqn_%d_sym'%self.n)
        self.ALL_STATES = bin_states(self.n,True)  # all 2^n possible binary states in {-1,1} basis
        self.uniqP,self.uniqStates = state_probs(sample)
        self.uniqP /= self.uniqP.sum()
        self.lamda = lamda
        self.find_cond_states()
    
    def calc_grad(self,eps=1e-8,center=True,weights=None):
        """Calculate gradient to error function (<<f_k>>-<f_k>)^2"""
        if weights is None:
            weights = np.ones_like(self.lamda)
        grad = np.zeros(len(self.lamda))
        if center:
            for i in xrange(len(self.lamda)):
                fkData1,fkMaxent1 = self.perturb(eps,i)
                fkData0,fkMaxent0 = self.perturb(-eps,i)
                grad[i] = ( np.sqrt( weights.dot((fkData1-fkMaxent1)**2) ) 
                          - np.sqrt( weights.dot((fkData0-fkMaxent0)**2) ) )/(2*eps)
        else:
            for i in xrange(len(self.lamda)):
                fkData1,fkMaxent1 = self.perturb(eps,i)
                fkData0,fkMaxent0 = self.perturb(0,i)
                grad[i] = ( np.sqrt( weights.dot((fkData1-fkMaxent1)**2) ) 
                          - np.sqrt( weights.dot((fkData0-fkMaxent0)**2) ) )/eps
        return grad
    
    def error(self,weights=None):
        p = self.ising_eqns.p(self.lamda)
        if weights is None:
            return np.linalg.norm( self.fk_data(p)-self.fk_maxent(p) )
        return np.sqrt( weights.dot( (self.fk_data(p)-self.fk_maxent(p))**2 ) )

    def perturb(self,eps,i):
        """
        Parameters
        ----------
        i : int
        eps : float
        """
        self.lamda[i] += eps
        p = self.ising_eqns.p(self.lamda)

        # Calculate the data term <<f_k>>.
        fkData = self.fk_data(p)

        # Calculate the model average <f_k>
        fkMaxent = self.fk_maxent(p)

        self.lamda[i] -= eps
        return fkData,fkMaxent

    def fk_data(self,p):
        """
	Calculate the data term <<f_k>>.
        Calculate whole set of observables for each unique incomplete state in sample.
	"""
        for si,s in enumerate(self.uniqStates):
            if si==0:
                firstrow = self.conditional_observable(s,p)
                fkData = np.zeros((len(self.uniqStates),len(firstrow)))
                fkData[0] = firstrow
            else:
                fkData[si] = self.conditional_observable(s,p)
        # Weighted average
        return fkData.T.dot(self.uniqP)

    def fk_maxent(self,p):
        # Calculate the model average <f_k>
        return p.dot( calc_observables(self.ALL_STATES) )

    def conditional_observable(self,s,p):
        """
        Calculate the observables in the distribution where the nonzero elements of s 
        are fixed.

        Parameters
        ----------
        s : ndarray
            Particular incomplete state of interest.
        p : ndarray
            Full probability distribution of entire system.
        """
	for s_,ix in self._cond_sample_ix:
            if (s_==s).all():
                return p[ix].dot(calc_observables(self.ALL_STATES[ix]))/p[ix].sum()
        raise Exception("State not found. This should not happen if contained in uniqStates.")

    def find_cond_states(self):
    	"""
	Find all the states that correspond to having a few spins fixed in self.sample. Find and
	store these so that we don't have to look for them over and over again.
	"""
	self._cond_sample_ix = []  # indices for corresponding samples, tuple of state and the
				   # indices
	for  s in self.uniqStates:
	    emptyix = s==0
	    nEmpty = emptyix.sum()
	    fillstates = bin_states(nEmpty,True)

	    # Iterate through all possible states that could fill blanks, such that we hold
	    # the nonzero spins fixed.
	    ix = []  # index of states that belong to conditional distribution
	    for i,s_ in enumerate(fillstates):
		s[emptyix] = s_
		ix.append( np.where((s==self.ALL_STATES).all(1))[0][0] )
	    s[emptyix] = 0

	    self._cond_sample_ix.append( (s,ix) )
#end Tester


# MCH estimate of the gradient of the error.
class TesterMCH(object):
    def __init__(self,sample,lamda):
        """
        Parameters
        ----------
        sample : ndarray
            Sample of data.
        lamda : ndarray
            parameters
        """
        self.uniqP,self.uniqStates = state_probs(sample)
        self.uniqP /= self.uniqP.sum()
        self.lamda = lamda
        
        self.calc_e = lambda sample,lamda : -calc_observables(sample).dot(lamda)
        self.N = sample.shape[1]
        self.initialize_sampler()
        
    def initialize_sampler(self):
        self.sampler = MCIsing(self.N,self.lamda,self.calc_e)
        self.generate_sample()
        
        # Sample from the conditional distributions.
        self.generate_cond_sample()
    
    #def average_grad(self,n_iters=5):
    #    grad = np.zeros((self.lamda))
    #    for i in xrange(n_iters):
    #        

    def calc_grad(self,eps=1e-4):
        """Calculate gradient to error function (<<f_k>>-<f_k>)^2"""
        self.generate_sample()
        # Sample from the conditional distributions.
        self.generate_cond_sample()
        
        # Calculate gradient using MCH.
        grad = np.zeros(len(self.lamda))
        for i in xrange(len(self.lamda)):
            fkData1,fkMaxent1 = self.perturb(eps,i)
            fkData0,fkMaxent0 = self.perturb(-eps,i)
            grad[i] = ( np.sqrt( ((fkData1-fkMaxent1)**2).sum() )
                      - np.sqrt( ((fkData0-fkMaxent0)**2).sum() ) )/(2*eps)
        return grad
        
    def perturb(self,eps,i):
        """
        Different routines must be used for <<<f_k>> compared with <f_k>.
        
        Parameters
        ----------
        eps : float
        i : int
        """
        # Calculate the data term <<f_k>>.
        fkData = self.fk_data(eps,i)

        # Calculate the model average <f_k>
        fkMaxent = self.fk_maxent(eps,i)
        
        return fkData,fkMaxent
    
    def error(self):
        return np.linalg.norm( self.fk_data(0)-self.fk_maxent(0) )

    def fk_data(self,eps,ix=0):
        """
        Calculate the data term <<f_k>>.
        
        Given the perturbation to make on the parameters, predict how the distribution
        changes.
        """
        # Calculate whole set of observables for each unique incomplete state in sample.
        # Predict new fk using MCH approximation.
        for si,sample in enumerate(self.condSamples):
            if si==0:
                firstrow = self.conditional_observable(sample,eps,ix)
                fkData = np.zeros((len(self.uniqStates),len(firstrow)))
                fkData[0] = firstrow
            else:
                fkData[si] = self.conditional_observable(sample,eps,ix)
        # Weighted average
        return fkData.T.dot(self.uniqP)

    def fk_maxent(self,eps=0,i=0):
        """Calculate the model average."""
        return self.conditional_observable(self.sample,eps,i)

    def conditional_observable(self,sample,eps=0,ix=0):
        """
        Calculate the observables in the distribution where the nonzero elements of s 
        are fixed.
        
        If eps!=0, then use MCH to make a prediction of how the distribution changes to
        estimate the observable.

        Parameters
        ----------
        sample : ndarray
        eps : float,0
        ix : int,0
            index of parameter to perturb by eps
        """
        if eps==0:
            return calc_observables(sample).mean(0)
        
        dlamda = np.zeros(len(self.lamda))
        dlamda[ix] += eps
        return mch_approximation(sample,dlamda)
    
    def generate_sample(self,sample_size=100):
        self.sampler.theta = self.lamda
        self.sampler.generate_samples_parallel(sample_size,
                                               n_iters=100,
                                               cpucount=4)
        self.sample = self.sampler.samples
    
    def generate_cond_sample(self,n_sample=100):
        self.sampler.theta = self.lamda
        
        self.condSamples = []
        for s in self.uniqStates:
            fixedIx = np.where(s!=0)[0].tolist()
            frozenSpins = [(ix,s[ix]) for ix in fixedIx]
            sample,E = self.sampler.generate_cond_samples(n_sample,
                                                          frozenSpins,
                                                          burn_in=50)
            self.condSamples.append( sample )
#end testerMCH



def descend_gradient(tester,eps=1e-2,inertia_weight=.3,n_iter=10):
    """
    Gradient descent with inertia. A kind of smoothing.
    
    Parameters
    ----------
    tester : class instance
        Class with .error(), .calc_grad() methods and .lamda field.
    eps : float
    inertia_weight : float
    n_iter : int
    
    Returns
    -------
    errorHistory : list
    """
    assert 0<=inertia_weight<1
    errorHistory = []
    
    errorHistory.append( tester.error() )
    prevStep = np.zeros_like(tester.lamda)
    for i in xrange(n_iter):
        # Take a step down the gradient.
        grad = tester.calc_grad()
        nextStep = -grad*eps + inertia_weight*prevStep
        
        tester.lamda += nextStep
        errorHistory.append( tester.error() )
    return errorHistory

def conditional_likelihood(sample,lamda):
    """
    Likelihood of incomplete data given the parameters.
    
    Parameters
    ----------
    sample : ndarray
        Sample of incomplete data.
    lamda : ndarray
        Ising model parameters.
    """
    n = sample.shape[1]
    ising_eqns = importlib.import_module('ising_eqn_%d_sym'%n)
    ALL_STATES = bin_states(n,sym=True)
    p = ising_eqns.p(lamda)
    likelihood = 0
    
    for s in sample:
        zeroix = s==0
        sampleix = ( ALL_STATES[:,zeroix==0]==s[None,zeroix==0] ).all(1)
        likelihood += np.log( p[sampleix].sum() )
    return likelihood

