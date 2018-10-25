# Ising solver.
# 2015-04-30
# MIT License
# 
# Copyright (c) 2017 Edward D. Lee
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

import numpy as np
from misc_fcns import *
import workspace.utils as ws
import scipy.io as sio
from scipy.spatial.distance import squareform
import fast,sys,os
from joblib import Parallel,delayed
import scipy.optimize as opt
from .samplers import MCIsing,WolffIsing,ParallelTempering
#import copy_reg
#import types
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def _testing_algorithm(n,etamch,maxdjnorm,maxerr,maxlearnsteps,djexpnormscalemx,
                      djexpnormscalemn,
                      name='',J0=None,alpha=.4,eta=1.,errscale=1.):
    """
        Test the Boltzmann learning algorithm with MCH speed up by putting in
        some random test J, generating some samples, and checking that those
        samples match up.
        n : number of spins
        ETA_MCH : scaling factor for relating dsisj to dJ
        name :
        errscale : std of random errors put on initial J
    2014-02-19
    """
    import time

    if not isinstance(name,str):
        try:
            name = str(name)
        except err:
            raise("name argument could not be converted to a string.")

    nn = n*(n-1)/2
    if J0 is None:
        J0 = np.random.randn(nn+n)

    t0 = time.time()
    samples = generate_samples(J0,n,1e4)
    print("Time to generate samples is "+str(time.time()-t0))
    sisj0 = calc_sisj(samples)

    J, exitflag = solve_J(sisj0,J0+np.random.normal(scale=errscale,size=nn+n),n,
                          errmx=maxerr,eta=eta,etamch=etamch,name=name,alpha=alpha,
                          maxlearnsteps=maxlearnsteps, maxdjnorm=maxdjnorm,
                          djexpnormscalemx=djexpnormscalemx, djexpnormscalemn=djexpnormscalemn)
    sisj = calc_sisj( generate_samples(J,n,1e4) )
    print(sisj.shape)
    print(sisj0.shape)
    print(exitflag)

    # Plotting.
#    import matplotlib.pyplot as plt
#    fig = plt.figure(figsize=[12,6])
#    ax1 = fig.add_subplot(121,
#                          xlabel=r'$<sisj>_{\rm exp}$',
#                          ylabel=r'$<sisj>$')
#    ax1.plot(sisj0,sisj,'o',[0,1],[0,1],'k-')
#    ax2 = fig.add_subplot(122,
#                          xlabel=r'$J_{\rm exp}$',
#                          ylabel=r'$J$')
#    ax2.plot(J0,J,'o')

    return (sisj0,J0,sisj,J)

class Solver():
    """ 
    Parent of the different methods for solving. Holds the common functions and attributes necessary to run the different solvers. The corresponding solvers are MonteCarloHistogram and Exact.

    To run this, you typically should instantiate this Solver, then call the update_solver() routine to set the particular solver. Then, solve.
    2015-05-05
    
    Params:
    -------
    J 
        Ising model parameters, lower order parameters are first
    N (int)
        size of system
    S 
        number of samples to take for Monte Carlo
    constraints 
        correlations to constrain model as concatenate((sisj,si))
    nJobs (4,int)
    display (False,bool)
        output status of solver
    solver (str)
        'mch' or 'exact'
    samples (ndarray)
        samples to start mch solver with
    E (float)
        energies of samples
    runStatus
    """
    def __init__(self,N,constraints,J=None,samples=None,S=1e3,E=None,nJobs=4,
                 display=False,solveOutput=None):
        self.J = J  # Ising model parameters, lower order parameters are first in constrast with the given constraints
        self.N = N  # size of system
        self.NN = N*(N-1)/2
        self.S = S  # number of samples to take
        self.constraints = constraints  # correlations to constrain model
        self.nJobs = nJobs
        self.solveOutput = solveOutput
        self.display = display
        self.solver = None
        self.runStatus = {}
        
        if samples is None:
            self.samples = np.random.randint(2,size=(S,N)).astype(int)
            self.sisj = None
            self.E = None
        else:
            self.samples = samples  # if any samples to jump start solver with
            self.sisj = self.calc_sisj(self.samples)
            self.E = fast.calc_e(self.J,self.samples)
        self.jac = None
        return

    def update_solver(self,solver):
        """
        Call the right solver to solve for the parameters.
        2015-05-05
        """
        if solver=='mch':
            self.solver = MonteCarloHistogram( self.N, self.constraints, self.J, 
                                               self.samples, self.S, self.E, 
                                               self.nJobs, self.display)
        elif solver=='exact':
            self.solver = ExactNonlinear(self.N,self.constraints,self.J,self.samples,self.S,self.E,self.nJobs,self.display)
        else:
            raise Exception("Invalid solver option.")

    # NOTE: add code to save the number of sample to reach equilibrium, most likely they
    # will be similar for similar hamiltonians
    def generate_samples(self,J=None,iters=None):
        """
        Generate samples via Metropolis method. Currently implemented as a loop. Can
        be implemented in parallel as in alternative equilibrate_samples code.
        2013-03-04

        Params:
        -------
        J (ndarray)
        iters (int)
        """
        if J is None:
            J = self.J

        if iters is None:
            # Pick 10 random self.samples from which to caclulate autocorrel.
            if self.S>10:
                samples = np.random.randint(2,size=(10,self.N))
            else:
                samples = np.random.randint(2,size=(1,self.N))

            if 2**self.N < 100:
                iters = self.get_iter_length(J,samples.copy(),iters=100)
            else:
                iters = self.get_iter_length(J,samples.copy())
            iters += 30 

        print("iters = %d" %iters)
        # Equilibrate each sample.
    #    (e,self.samples) = fast.equilibrate_samples(self.J,np.reshape(self.samples[0,:],(1,self.N)),iters,
    #                                           self.S,burnin=iters)
        #e,self.samples = fast.equilibrate_samples_rand_states(self.J,self.N,iters,self.S,burnin=iters)

        # Both these sampling algorithms work 2015-04-30
        self.equilibrate_samples_rand_start_p( iters, burnin=iters )
        #self.equilibrate_samples( iters, burnin=iters )
        return 

    def sample_gibbs(JMat,sample0,N):
        """
        Gibb's sampling. Not the current way of solving the problem.
        2013-03-05
        """
        randIx = np.random.randint(N)
        eUp = 0

        # Calculate energy when holding s_i up.
        sample0[0,randIx] = 1
        eUp = -np.sum( JMat[randIx,:]*sample0 )
        probUp = 1/(1+np.exp(eUp))

        # Return to si = 0 if not probable.
        if ( np.random.rand()>probUp ):
            sample0[0,randIx] = 0

        return sample0

    def sample_metropolis(self,JMat,sample0):
        """
        Metropolis sampling.
        Seems to return more accurate answers with J = [1,1,1] test case.

        sample0: must be a 2d array
        2014-01-27
        """
        randIx = np.random.randint(sample0.shape[1])
        sample0[0,randIx] = 1-sample0[0,randIx]
        
        # Remember that energy is defined as the negative.
        # Spin goes 1->0
        if sample0[0,randIx]==0:
            de = np.sum(JMat[randIx,:]*sample0) +JMat[randIx,randIx]
        # Spin goes 0->1
        else:
            de = -np.sum(JMat[randIx,:]*sample0) 
        
        # Only accept flip if dE<=0 or probability exp(-dE)
        # Thus reject flip if dE>0 and with probability (1-exp(-dE))
        if ( de>0 and (np.random.rand()>np.exp(-de)) ):
            sample0[0,randIx] = 1-sample0[0,randIx]
            return 0.
        else:
            return de

    def equilibrate_samples(self,iters,burnin=200):
        """
        2015-04-30
        """
        sample0 = np.random.randint(2,size=(1,self.N))
        e0 = fast.calc_e(self.J,sample0)
        JMat = squareform(self.J[self.N:])
        JMat[np.eye(self.N)==1] = self.J[:self.N]
        s = 0
        
        # First, burn in.
        for j in range(burnin):
            # sample0 = sample_gibbs(JMat,sample0,N)
            e0 += self.sample_metropolis(JMat,sample0)
        
        # Now, sample.
        while s<self.S:
            for j in range(int(iters)):
                # sample0 = sample_gibbs(JMat,sample0,N)
                e0 += self.sample_metropolis(JMat,sample0)
            self.samples[s] = sample0.copy()
            self.E[s] = e0
            s += 1
        return

    def equilibrate_samples_save_e_p(self,J,samples,iters):
        """
            Equilibrate using Metropolis, but save and return energy of each state using joblib for parallel processing.
        2014-05-06
        """
        (S,N) = samples.shape
        NN = N*(N-1)/2
        JMat = fast.convert_utri_to_array(J[N:],J[:N],N)
        
        results = list(zip(* Parallel(n_jobs=self.nJobs)(delayed(fast.sample_f_save_e)(s,samples[s],JMat,J,iters)
                                                        for s in range(S)) ))
        save_e,samples = np.array(results[0]),np.array(samples)
        return (save_e,samples)

    def equilibrate_samples_rand_start_p(self, iters, burnin=200 ):
        """
        Sample state space starting at a random point every time. This is a more efficient
        sampling of the space when we have a large spin system.
        2014-05-06
        """
        samples = np.random.randint(2,size=(self.S,self.N))
        JMat = fast.convert_utri_to_array(self.J[self.N:],self.J[:self.N],self.N)

        results = list(zip(* Parallel(n_jobs=self.nJobs)(delayed(fast.sample_f)(s,samples[s],JMat,self.J,iters,burnin)
                                            for s in range(int(self.S))) ))
        self.E = np.array(results[0])
        self.samples = np.array(results[1])
        return 

    def calc_e(self):
        """
        Works with either {0,1} or {-1,1}.
        2015-04-30
        """
        # if only one sample was given must get dimensions differently from vector
        if self.samples.ndim==1:
            S = 1
            self.samples = np.reshape(self.samples,(1,self.N))
        else:
            S = self.samples.shape[0]

        # initialize
        e = np.zeros((S))

        for s in range(S):
            k = 0
            for i in range(self.N):
                e[s] += self.J[i]*self.samples[s,i]
                for j in range(i+1,self.N):
                    e[s] += self.J[k+self.N]*self.samples[s,i]*self.samples[s,j]
                    k+=1
        return -e
    
    def calc_sisj(self,data=None):
        """
        2015-05-01
        """
        if data is None:
            data = self.samples
        sisj = np.zeros(self.N*(self.N-1)/2)

        k=0
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                sisj[k] = np.mean(data[:,i]*data[:,j])
                k+=1

        return np.concatenate((np.mean(data,0),sisj))

    def convert_params(h,J,order):
        """
            Convert parameters from 0,1 formulation to +/-1 and vice versa.
        2014-05-12
        """
        from entropy.fcns import squareform

        if len(J.shape)!=2:
            Jmat = squareform(J)
        else:
            Jmat = J
            J = squareform(J)
        
        if order==0:
            # Convert from 0,1 to -/+1
            Jp = J/4.
            hp = h/2 + np.sum(Jmat,1)/4.
        elif order==1:
            # Convert from -/+1 to 0,1
            hp = 2.*(h - np.sum(Jmat,1))
            Jp = J*4.

        return (hp,Jp)
    
    def get_iter_length(self,J,samples,iters=1000 ):
        """
        Sample and get autocorrelation to get estimate for how many iters to run
        Metropolis for. Each recursive step doubles length by two.
        2013-03-04
        """
        if 'f' not in globals():
            f = sys.stdout
    #     f.write("running get_iter_length "+str(iters)+'\n')
        (S,N) = samples.shape
        autocorrelE = np.zeros((S,iters/10))
        autocorrelE[:,0] = 1

        # Run Metropolis algorithm.
        (save_e,samples_) = self.equilibrate_samples_save_e_p(J,samples.copy(),iters)
    #    (save_e,samples_) = fast.equilibrate_samples_save_e(J,samples.copy(),iters)
    #    if iters==200:
    #        sio.savemat('test',{'save_e':save_e})
    #        raise Exception("Leaving.")

        # Get autocorrel of sequence of Metropolis states.
        for i in range(save_e.shape[0]):
            for j in range(1,int(iters/10)):
                autocorrelE[i,j] = pearson_corr(save_e[i,:-j:j],save_e[i,j::j])
    #     f.write("J = ")
    #     for i in range(15):
    #         f.write("%f, " %J[i])
    #     f.write("\n")
    #     f.write("mean E = ")
    #     for i in range(autocorrelE.shape[1]):
    #         f.write("%f, " %np.mean(save_e,0)[i])
    #     f.write("\n")
    #     if np.any(np.isnan(save_e)):
    #         f.write("Some energies are nan.\n")

        mAutocorrelE = np.mean(autocorrelE,0)
    #     f.write("mean = ")
    #     for i in range(mAutocorrelE.size):
    #         f.write("%f, " %mAutocorrelE[i])
    #     f.write("\n")
        stdAutocorrelE = np.std(autocorrelE,0)/np.sqrt(S)
    #     f.write("std = ")
    #     for i in range(mAutocorrelE.size):
    #         f.write("%f, " %stdAutocorrelE[i])
    #     f.write("\n")

        # Check if iterations were sufficient to reach decorrelation.
        # If not, must increase number of iterations.
        # Plot autocorrel descent.
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.errorbar(range(iters/10),mAutocorrelE,stdAutocorrelE)
        # fig.savefig('autocorrel'+str(iters)+'.png',bbox_inches='tight')
    #     ix = np.where( (mAutocorrelE-stdAutocorrelE/2)<0 )[0]
        ix = np.where( (mAutocorrelE-stdAutocorrelE/2)<0 +
                        np.isclose(0,mAutocorrelE,atol=1e-2) )[0]
        if not ix.any():
            ix = self.get_iter_length(J,samples,iters*2)

        if ix.size>1:
            ix = ix[0]
        return ix

# ------------------------------------------------------------------------------------
class MonteCarloHistogram(Solver):
    """
    Child of model class that contains methods specific to solving the Ising model using MCH methods.

    There are two ways of solving the problem, gradient descent and straight up MCH. 
    1) Gradient descent estimates the local Jacobian using the sampled at the current parameters. This probably does best with very high sampling for good accuracy (these solvers aren't designed for noisy gradient estimates). 
    2) MCH is the Broderick algorithm with many parameters to tweak. It may take several tries with changing parameter values to get the solution to converge. I did not implement a dynamically changing eta, for example, because that would make things much more complicated. Overall, this method involves some hand-holding.
    2015-05-05
    """
    def __init__(self,*args):
        Solver.__init__(self,*args)
        return
    
    def solve(self,J0=None,solver='gd',**kwargs):
        if J0 is None:
            J0 = self.J
        if solver=='gd':
            self.gradient_descent(J0,**kwargs)
        elif solver=='mch':
            return self.MCH(J0,**kwargs)
        else:
            raise Exception("Not a valid option for solver in MonteCarloHistogram.")

    def gradient_descent(self, J0=None, iters=None, **kwargs):
        """
        Approximate the Jacobian using MCH approximation and feed that into a least squares solver.

        Params:
        -------
        sisj 
            vector of correlations where the first n(n-1)/2 are <sisj> and last n are <si>
        J0
            vector of initial parameter guesses in same order as sisj
        N : size of system
        2015-05-01
        """
        import scipy.optimize as sopt
        import entropy.entropy as entropy

        sisj = self.constraints
        if J0 is None:
            J0 = self.J

        # Function to find zeros of.
        def f(J):
            if np.any(np.abs(J)>10):
                return np.zeros((J.size))+1e60
            #if sym: 
            #    J = np.hstack(( J,-.5*np.sum(squareform(J),1) ))
            
            print("Sampling...") 
            if iters is None:
                self.generate_samples()
            else:
                self.generate_samples(iters=iters)
            self.sisj = self.calc_sisj(self.samples)
            return self.sisj-self.constraints

        def jacFun(eps): 
            print("Computing Jacobian...")
            self.jac = self.get_jac_p( self.sisj, self.samples, eps=eps )
            return self.jac
        
        #Jf = sopt.fsolve( f,J0,fprime=load_temp_jac,full_output=True,xtol=1e-10 )
        self.solverOutput = sopt.leastsq(f, J0, Dfun=lambda x: jacFun(1e-2),
                        factor=.1, full_output=True, epsfcn=1e-3, ftol=.01,
                        col_deriv=True, **kwargs )
        self.J = self.solverOutput[0]
        return

    def get_jac_p(self,sisj,samples, eps=1e-2,sym=False):
        """
        Call get_jac_col() using parallel for loop.
        2014-05-06
        """
        if type(eps) is float:
            eps = np.zeros((sisj.size))+eps

        jacT = np.array( Parallel(n_jobs=self.nJobs)(delayed(get_jac_col)(i,sisj,samples,eps,sym=sym) 
                                                 for i in range(sisj.size)) )
        return jacT

    def get_jac(self,sisj,samples,eps=1e-2):
        """
        Return Jacobian estimate using MCH algorithm. The dJ used will is given as eps
        but this algorithm will adaptively adjust the interval depending on whether
        floating point precision will be good enough to estimate that interval.
        2015-05-01
        """
        jac = np.zeros((sisj.size,sisj.size))
        for i in range(sisj.size):
            dJ = np.zeros((sisj.size))
            dJ[i] = eps
            sisj_,Zfracisbad = sample_MCH(dJ,samples)
            while Zfracisbad:
                warnings.warn("Z frac is bad. Dynamically changing eps value.")
                eps /= 10.
                dJ[i] = eps
                sisj_,Zfracisbad = sample_MCH(dJ,samples)
                if eps<1e-6:
                    raise Exception("eps very small.")
                print("eps = %1.3f" %eps)
            jac[i,:] = (sisj_-sisj)/eps
        return jac
    
    def _sample_MCH(dJ,samples):
        """
            Sample Monte Carlo histogram.
            One problem here is that the calculated fraction of partition functions increases
            exponentially with the energy so it can easily become to large or too small for
            Python to deal with. In those cases, we automatically scale the given dJ in the
            right direction (positive or negative).
        2014-01-18
        """
        N = samples.shape[1]
        NN = N*(N-1)/2
        sisj = np.zeros((NN+N))
        ZFrac = np.float128(0.) # make sure we have high accuracy
        Zfracisbad = False
        if 'f' not in globals():
            import sys
            f = sys.stdout

        # At limits of extrapolation, ZFrac will become very, very small.
        expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
        ZFrac = np.mean( expE )
        while (ZFrac < 1e-30) or (ZFrac > 1e30):
            dJ /= 2.
            expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
            ZFrac = np.mean( expE )
            Zfracisbad = True
            f.write("ZFrac too small/large for good estimate so dividing dJ.\n")
    #     f.write("ZFrac = "+str(ZFrac)+"\n")

        k = 0
        for i in range(N-1):
            for j in range(i+1,N):
                sisj[k] = np.mean( samples[:,i]*samples[:,j]*expE )
                k +=1

        for i in range(N):
            sisj[k] = np.mean( samples[:,i]*expE )
            k += 1

        sisj /= float(ZFrac)

        return sisj,Zfracisbad

    def get_eta(distNorm,eta=np.nan,sampleErrTol=3e-2,minEtaChange=1e-5):
        """
        2013-03-18
        """
        return (1,0)

        dErr = distNorm[-1]-distNorm[-2]

        if (endIx==0):
            return (standard_eta_ret(distNorm[endIx]),0)

        # Two main possibilities
        # 1. error shrinks,
        #   a. ok
        #   b. too slowly (<= 1e-3), unless error is already quite small
        # 2. error grows, error stays the same
        #   a. error grows acceptably perhaps due to sampling errors (5e-2)
        #   b. error grows unacceptably and we decrease eta, unless ETA is already
        #       very small
        if (dErr<0):
            if (np.abs(dErr)>5e-3):
                return (eta,0)
            else:
                f.write("Growing ETA by 1.25 to "+str(eta*1.25)[:5]+
                        " given dErr = "+str(dErr)[:15]+"\n")
                return (eta*1.25,0)
        else:
            if eta>minEtaChange:
                if (distNorm[endIx]/distNorm[endIx-1]<1.2): #and (dErr<sampleErrTol):
                    return (eta,0)
                else:
                    f.write("Shrinking ETA by 2...\n")
                    return (eta/2.0,1)
            else:
                # return (0,0)
                return (standard_eta_ret(distNorm[endIx]),0)

    def standard_eta_ret(distNorm):
        """
        2013-03-12
        """
        return 1

        if distNorm>=1:
            return 1.0
        elif distNorm >= .5:
            return .1
        elif distNorm >= .1:
            return .01
        elif distNorm >= .001:
            return .001
        else:
            return .001


    def _get_jac(sisj,samples,eps=1e-2):
        """
            Return Jacobian estimate using MCH algorithm. The dJ used will is given as eps
            but this algorithm will adaptively adjust the interval depending on whether
            floating point precision will be good enough to estimate that interval.
            2014-02-21
        """

        jac = np.zeros((sisj.size,sisj.size))
        for i in range(sisj.size):
            dJ = np.zeros((sisj.size))
            dJ[i] = eps
            sisj_,Zfracisbad = sample_MCH(dJ,samples)
            while Zfracisbad:
                warnings.warn("Z frac is bad. Dynamically changing eps value.")
                eps /= 10.
                dJ[i] = eps
                sisj_,Zfracisbad = sample_MCH(dJ,samples)
                if eps<1e-6:
                    raise Exception("eps very small.")
                print("eps = %1.3f" %eps)

            jac[:,i] = (sisj_-sisj)/eps
        return jac


    def _sample_MCH(dJ,samples):
        """
            Sample Monte Carlo histogram.
            One problem here is that the calculated fraction of partition functions increases
            exponentially with the energy so it can easily become to large or too small for
            Python to deal with. In those cases, we automatically scale the given dJ in the
            right direction (positive or negative).
        2014-01-18
        """
        N = samples.shape[1]
        NN = N*(N-1)/2
        sisj = np.zeros((NN+N))
        ZFrac = np.float128(0.) # make sure we have high accuracy
        Zfracisbad = False
        if 'f' not in globals():
            import sys
            f = sys.stdout

        # At limits of extrapolation, ZFrac will become very, very small.
        expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
        ZFrac = np.mean( expE )
        while (ZFrac < 1e-30) or (ZFrac > 1e30):
            dJ /= 2.
            expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
            ZFrac = np.mean( expE )
            Zfracisbad = True
            f.write("ZFrac too small/large for good estimate so dividing dJ.\n")
    #     f.write("ZFrac = "+str(ZFrac)+"\n")

        k = 0
        for i in range(N-1):
            for j in range(i+1,N):
                sisj[k] = np.mean( samples[:,i]*samples[:,j]*expE )
                k +=1

        for i in range(N):
            sisj[k] = np.mean( samples[:,i]*expE )
            k += 1

        sisj /= float(ZFrac)

        return sisj,Zfracisbad

    

    def iterate_MCH(self,dJ,distNorm,**kwargs):
        """
        Run histogram sampling. Given the current sample of states from the current J parameters, find the appropriate dJ to get closest to the data while keeping extrapolations within a reasonable error limit (by not extrapolating with too large of dJ).

        Get max dJ norm value from MAX_DJ_NORM
        2015-05-01

        Params:
        --------
        dJ (ndarray)
        distNorm (list)
            norm of dsisj
        **kwargs
        """
        MCHistCount = 0
        learnSteps = 0
        resetcount = 0
        prevdJ = dJ.copy()
        dsisj = self.constraints - self.sisj

        while (np.abs(dJ).max()<kwargs['MAX_DJ']) and (np.linalg.norm(dJ)<kwargs['MAX_DJ_NORM']) and \
              (MCHistCount<kwargs['MAX_MCH_COUNT']):
            self.printout('MC hist sample '+str(MCHistCount)+'\n')
            
            # Sample from dJ.
            nextsisj,Zfracisbad = sample_MCH( dJ, self.samples )
            dsisj = self.constraints - nextsisj
            distNorm.append( np.linalg.norm(dsisj) )

            if Zfracisbad:
                # Delete previous norm calculation and restore previous dJ.
                distNorm.pop();
                return learnSteps, dsisj, dJ

            # Save results and set up next iteration.
            # test whether dJ is too large.
            prevdJ = dJ.copy()
            dJ += dsisj * kwargs['ETA_MCH']
            self.sisj = nextsisj

            #self.printout('\tdist = '+str(distNorm[-1])+'\n')
            MCHistCount +=1
            learnSteps += 1
        return learnSteps, dsisj, dJ

    def printout(self,s):
        if self.display:
            print(s)
    
    def MCH(self,J0=None, 
                tol=2e-2, errnormtol=.01, samplesn=1e4, maxmchcount=10,
                maxlearnsteps=200, eta=1., etamch=.5, name='', alpha=0,
                maxdjnorm=None,
                maxdJ=.5,
                samplestepsmx=15,
                display=False):
        """
        Solve for the parameters Jij and hi in a pairwise maximum entropy model given the
        average pairwise correlations <sisj> and means <si> using just the MCH sampling algorithm. This is ideal for when we have a good approximation to the real solution and need to close the gap because this takes a long time to converge accurately (although theoretically it can converge exactly).

        maxdjnorm : the norm change allowed in J per iteration, max value at 1
            and below that is proportional to the error
        maxmchcount : max no. of consecutive iterations with MC histogram
            approx.
        alpha : weight given to inertial term when updating parameters
        errmx : max deviation allowed for any single fit to corresonding statistic
        errnormmx : tolerated error for norm of vector difference to statistics
        2015-08-15
        """
        if not J0 is None:
            self.J = J0

        # Solver settings.
        settings = {'MAX_LEARN_STEPS':maxlearnsteps, 'MAX_MCH_COUNT':maxmchcount, 
                    'TOL':tol,'NORMTOL':errnormtol,
                    'ALPHA':alpha, 'ETA':eta, 'ETA_MCH':etamch, 
                    'MAX_DJ_NORM':np.sqrt(self.N),
                    'MAX_DJ':maxdJ,
                    'SAMPLE_STEPS_MX':samplestepsmx}

        # Initialize variables.
        allJ = np.zeros((maxlearnsteps+1,self.J.size))
        allsisj = np.zeros((maxlearnsteps+1,self.J.size))
        distNorm = [np.inf,]
        dJ = 0.
        sisjerrs = [] # keep track of vector of differences from estimated correlation to
                        # real correlations

        # Initialize counters and switches.
        self.runStatus = {'learnSteps':0,'sampleSteps':0,'mcHistCount':0}
        #learnSteps: total number of iterations of J that have happened  including MC hist and Metropolis sampling
        #samplesteps  number of times Metropolis sampling has happened
        #SIncremented boolean for whether S has been grown for error reduction

        # Get initial sample.
        self.generate_samples()
        self.sisj = self.calc_sisj()
        dsisj = self.constraints - self.sisj
        distNorm.append( np.linalg.norm(dsisj) )
        allJ[0,:], allsisj[0,:] = J0, self.sisj
        self.printout(np.max(np.abs(dsisj))) 
        
        # Solve for parameters J.
        while (np.max(np.abs(dsisj)) > settings['TOL'] and np.linalg.norm(dsisj) > settings['NORMTOL']) and \
                (self.runStatus['learnSteps'] <= settings['MAX_LEARN_STEPS']) and\
                (self.runStatus['sampleSteps'] <= settings['SAMPLE_STEPS_MX']):
            sisjerrs.append(dsisj)
            dJ = (dsisj*settings['ETA'] +settings['ALPHA']*dJ) * min([distNorm[-1],1.])
            
            # MCH prediction of change in J.
            learnStepsDelta,dsisj,dJ = self.iterate_MCH(dJ,distNorm,**settings)
            self.runStatus['learnSteps'] += learnStepsDelta

            # Update variables.
            self.J += dJ
            prevdJ = dJ.copy()
            allJ[self.runStatus['sampleSteps']+1,:] = self.J
            prevdsisj = dsisj.copy()

            # If MC histogram has gotten us close enough, take it.
            # MCH cannot return fits to each individual statistic and can
            self.printout("Max deviation is %f. Max allowed is %f.\n" 
                          %(np.max(np.abs(dsisj)),settings['TOL']) )
            if np.max(np.abs(dsisj)) < settings['TOL']:
                break

            # Metropolis sample.
            self.printout('\nSampling...\n')
            self.generate_samples()
            self.sisj = self.calc_sisj()
            dsisj = self.constraints - self.sisj
            distNorm.append( np.linalg.norm(dsisj) )
            self.printout("Max deviation is %f\n" % np.max(np.abs(dsisj)) )
            
            # Output
            self.printout('learnSteps='+str(self.runStatus['learnSteps'])+'\n')
            self.printout('\tdist = '+str(distNorm[-1])+'\n')
            self.printout('dJNorm = '+str(np.linalg.norm(dJ))+'\n')
            self.printout('distNorm = '+str(distNorm)+'\n')

            allsisj[self.runStatus['sampleSteps'],:] = self.sisj
    #        if not (learnSteps%10):
    #        sio.savemat('temp_J',{'allJ_hat':allJ,'distNorm':distNorm,
    #                'allsisj':allsisj,'J':J})
            #sio.savemat('sisjerrs'+name,{'sisjerrs':sisjerrs})
            self.runStatus['learnSteps'] += 1

        # Exitflag steps.
        exitflag = 0
        if (np.max(np.abs(dsisj)) < settings['TOL']) and \
                not (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 0
        elif not (np.max(np.abs(dsisj)) < settings['TOL']) and \
                    (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 1
        elif (np.max(np.abs(dsisj)) < settings['TOL']) and \
                (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 2
        if self.runStatus['sampleSteps']>settings['SAMPLE_STEPS_MX']:
            exitflag = 3
            print("MC sampling steps exceed allowed max.")
        self.printout("Done")
        return exitflag

# -------------------------------------------------------------------------------------------------
class _MonteCarloHistogram(Solver):
    """
    Child of model class that contains methods specific to solving the Ising model using MCH methods.

    There are two ways of solving the problem, gradient descent and straight up MCH. 
    1) Gradient descent estimates the local Jacobian using the sampled at the current parameters. This probably does best with very high sampling for good accuracy (these solvers aren't designed for noisy gradient estimates). 
    2) MCH is the Broderick algorithm with many parameters to tweak. It may take several tries with changing parameter values to get the solution to converge. I did not implement a dynamically changing eta, for example, because that would make things much more complicated. Overall, this method involves some hand-holding.
    2015-05-05
    """
    def __init__(self,*args):
        Solver.__init__(self,*args)
        return
    
    def solve(self,J0=None,solver='gd',**kwargs):
        if J0 is None:
            J0 = self.J
        if solver=='gd':
            self.gradient_descent(J0,**kwargs)
        elif solver=='mch':
            self.MCH(J0,**kwargs)
        else:
            raise Exception("Not a valid option for solver in MonteCarloHistogram.")

    def gradient_descent(self, J0=None, iters=None, **kwargs):
        """
        Approximate the Jacobian using MCH approximation and feed that into a least squares solver.

        Params:
        -------
        sisj 
            vector of correlations where the first n(n-1)/2 are <sisj> and last n are <si>
        J0
            vector of initial parameter guesses in same order as sisj
        N : size of system
        2015-05-01
        """
        import scipy.optimize as sopt
        import entropy.entropy as entropy

        sisj = self.constraints
        if J0 is None:
            J0 = self.J

        # Function to find zeros of.
        def f(J):
            if np.any(np.abs(J)>10):
                return np.zeros((J.size))+1e60
            #if sym: 
            #    J = np.hstack(( J,-.5*np.sum(squareform(J),1) ))
            
            print("Sampling...") 
            if iters is None:
                self.generate_samples()
            else:
                self.generate_samples(iters=iters)
            self.sisj = self.calc_sisj(self.samples)
            return self.sisj-self.constraints

        def jacFun(eps): 
            print("Computing Jacobian...")
            self.jac = self.get_jac_p( self.sisj, self.samples, eps=eps )
            return self.jac
        
        #Jf = sopt.fsolve( f,J0,fprime=load_temp_jac,full_output=True,xtol=1e-10 )
        self.solverOutput = sopt.leastsq(f, J0, Dfun=lambda x: jacFun(1e-2),
                        factor=.1, full_output=True, epsfcn=1e-3, ftol=.01,
                        col_deriv=True, **kwargs )
        self.J = self.solverOutput[0]
        return

    def get_jac_p(self,sisj,samples, eps=1e-2,sym=False):
        """
        Call get_jac_col() using parallel for loop.
        2014-05-06
        """
        if type(eps) is float:
            eps = np.zeros((sisj.size))+eps

        jacT = np.array( Parallel(n_jobs=self.nJobs)(delayed(get_jac_col)(i,sisj,samples,eps,sym=sym) 
                                                 for i in range(sisj.size)) )
        return jacT

    def get_jac(self,sisj,samples,eps=1e-2):
        """
        Return Jacobian estimate using MCH algorithm. The dJ used will is given as eps
        but this algorithm will adaptively adjust the interval depending on whether
        floating point precision will be good enough to estimate that interval.
        2015-05-01
        """
        jac = np.zeros((sisj.size,sisj.size))
        for i in range(sisj.size):
            dJ = np.zeros((sisj.size))
            dJ[i] = eps
            sisj_,Zfracisbad = sample_MCH(dJ,samples)
            while Zfracisbad:
                warnings.warn("Z frac is bad. Dynamically changing eps value.")
                eps /= 10.
                dJ[i] = eps
                sisj_,Zfracisbad = sample_MCH(dJ,samples)
                if eps<1e-6:
                    raise Exception("eps very small.")
                print("eps = %1.3f" %eps)
            jac[i,:] = (sisj_-sisj)/eps
        return jac
    
    def _sample_MCH(dJ,samples):
        """
            Sample Monte Carlo histogram.
            One problem here is that the calculated fraction of partition functions increases
            exponentially with the energy so it can easily become to large or too small for
            Python to deal with. In those cases, we automatically scale the given dJ in the
            right direction (positive or negative).
        2014-01-18
        """
        N = samples.shape[1]
        NN = N*(N-1)/2
        sisj = np.zeros((NN+N))
        ZFrac = np.float128(0.) # make sure we have high accuracy
        Zfracisbad = False
        if 'f' not in globals():
            import sys
            f = sys.stdout

        # At limits of extrapolation, ZFrac will become very, very small.
        expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
        ZFrac = np.mean( expE )
        while (ZFrac < 1e-30) or (ZFrac > 1e30):
            dJ /= 2.
            expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
            ZFrac = np.mean( expE )
            Zfracisbad = True
            f.write("ZFrac too small/large for good estimate so dividing dJ.\n")
    #     f.write("ZFrac = "+str(ZFrac)+"\n")

        k = 0
        for i in range(N-1):
            for j in range(i+1,N):
                sisj[k] = np.mean( samples[:,i]*samples[:,j]*expE )
                k +=1

        for i in range(N):
            sisj[k] = np.mean( samples[:,i]*expE )
            k += 1

        sisj /= float(ZFrac)

        return sisj,Zfracisbad

    def get_eta(distNorm,eta=np.nan,sampleErrTol=3e-2,minEtaChange=1e-5):
        """
        2013-03-18
        """
        return (1,0)

        dErr = distNorm[-1]-distNorm[-2]

        if (endIx==0):
            return (standard_eta_ret(distNorm[endIx]),0)

        # Two main possibilities
        # 1. error shrinks,
        #   a. ok
        #   b. too slowly (<= 1e-3), unless error is already quite small
        # 2. error grows, error stays the same
        #   a. error grows acceptably perhaps due to sampling errors (5e-2)
        #   b. error grows unacceptably and we decrease eta, unless ETA is already
        #       very small
        if (dErr<0):
            if (np.abs(dErr)>5e-3):
                return (eta,0)
            else:
                f.write("Growing ETA by 1.25 to "+str(eta*1.25)[:5]+
                        " given dErr = "+str(dErr)[:15]+"\n")
                return (eta*1.25,0)
        else:
            if eta>minEtaChange:
                if (distNorm[endIx]/distNorm[endIx-1]<1.2): #and (dErr<sampleErrTol):
                    return (eta,0)
                else:
                    f.write("Shrinking ETA by 2...\n")
                    return (eta/2.0,1)
            else:
                # return (0,0)
                return (standard_eta_ret(distNorm[endIx]),0)

    def standard_eta_ret(distNorm):
        """
        2013-03-12
        """
        return 1

        if distNorm>=1:
            return 1.0
        elif distNorm >= .5:
            return .1
        elif distNorm >= .1:
            return .01
        elif distNorm >= .001:
            return .001
        else:
            return .001


    def _get_jac(sisj,samples,eps=1e-2):
        """
            Return Jacobian estimate using MCH algorithm. The dJ used will is given as eps
            but this algorithm will adaptively adjust the interval depending on whether
            floating point precision will be good enough to estimate that interval.
            2014-02-21
        """

        jac = np.zeros((sisj.size,sisj.size))
        for i in range(sisj.size):
            dJ = np.zeros((sisj.size))
            dJ[i] = eps
            sisj_,Zfracisbad = sample_MCH(dJ,samples)
            while Zfracisbad:
                warnings.warn("Z frac is bad. Dynamically changing eps value.")
                eps /= 10.
                dJ[i] = eps
                sisj_,Zfracisbad = sample_MCH(dJ,samples)
                if eps<1e-6:
                    raise Exception("eps very small.")
                print("eps = %1.3f" %eps)

            jac[:,i] = (sisj_-sisj)/eps
        return jac


    def _sample_MCH(dJ,samples):
        """
            Sample Monte Carlo histogram.
            One problem here is that the calculated fraction of partition functions increases
            exponentially with the energy so it can easily become to large or too small for
            Python to deal with. In those cases, we automatically scale the given dJ in the
            right direction (positive or negative).
        2014-01-18
        """
        N = samples.shape[1]
        NN = N*(N-1)/2
        sisj = np.zeros((NN+N))
        ZFrac = np.float128(0.) # make sure we have high accuracy
        Zfracisbad = False
        if 'f' not in globals():
            import sys
            f = sys.stdout

        # At limits of extrapolation, ZFrac will become very, very small.
        expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
        ZFrac = np.mean( expE )
        while (ZFrac < 1e-30) or (ZFrac > 1e30):
            dJ /= 2.
            expE = np.float128( np.exp(-fast.calc_e(dJ,samples)) )
            ZFrac = np.mean( expE )
            Zfracisbad = True
            f.write("ZFrac too small/large for good estimate so dividing dJ.\n")
    #     f.write("ZFrac = "+str(ZFrac)+"\n")

        k = 0
        for i in range(N-1):
            for j in range(i+1,N):
                sisj[k] = np.mean( samples[:,i]*samples[:,j]*expE )
                k +=1

        for i in range(N):
            sisj[k] = np.mean( samples[:,i]*expE )
            k += 1

        sisj /= float(ZFrac)

        return sisj,Zfracisbad

    

    def iterate_MCH(self,dJ,distNorm,**kwargs):
        """
        Run histogram sampling. Given the current sample of states from the current J parameters, find the appropriate dJ to get closest to the data while keeping extrapolations within a reasonable error limit (by not extrapolating with too large of dJ).

        Get max dJ norm value from MAX_DJ_NORM
        2015-05-01

        Params:
        --------
        dJ (ndarray)
        distNorm (list)
            norm of dsisj
        **kwargs
        """
        RESET_COUNT_MX = 3 # times to allow resetting before killing loop

        MCHistCount = 0
        learnSteps = 0
        resetcount = 0
        prevdJ = dJ.copy()
        while MCHistCount<kwargs['MAX_MCH_COUNT'] and (resetcount<RESET_COUNT_MX):
            # Sample can increase or decrease error (with some margin for
            # error). In general, an increase will mean that we roll back the
            # step and re MCH. If decrease, then we accept the step and
            # accumulate changes in J.
            self.printout('MC hist sample '+str(MCHistCount)+'\n')
            
            # Sample from dJ.
            nextsisj,Zfracisbad = sample_MCH( dJ,self.samples )
            dsisj = self.constraints-nextsisj
            distNorm.append( np.linalg.norm(dsisj) )

            # Test whether sample is good given some erorr growth tolerance on the
            # norm as opposed to on the maximum error.
            if distNorm[-1] > (distNorm[-2] * kwargs['MCH_ERR_GROWTH_TOL']):
                self.printout("Error increasing. Leaving MCH iteration steps.\n")
                # Delete previous norm calculation and restore previous dJ.
                distNorm.pop();
                dJ = prevdJ.copy()
                break
            if Zfracisbad:
                MCHistCount = kwargs['MAX_MCH_COUNT']

            # Save results and set up next iteration.
            # test whether dJ is too large.
            prevdJ = dJ.copy()
            dJ += dsisj * kwargs['ETA_MCH']
            self.sisj = nextsisj

            while (np.linalg.norm(dJ)>kwargs['MAX_DJ_NORM_MCH']):
                dJ /= 2.
                self.printout("Decreasing dJ norm to "+str(np.linalg.norm(dJ))+"\n")
                resetcount += 1

            self.printout('\tdist = '+str(distNorm[-1])+'\n')
            MCHistCount +=1
            learnSteps += 1
        return learnSteps, dsisj, dJ

    def printout(self,s):
        if self.display:
            print(s)

    def sample_metropolis_dJ(self,dJ,dsisj,**settings):
        """
        Test whether the proposed dJ works by sampling Metropolis. If it increases the error, then adaptively adjust the dJ to see whether we can choose one that actually reduces it. Since the sampling step is really expensive, only try it a few times before giving up.

        1. Metropolis sample with gven parameters.
        2. Did the error increase or decrease? If it increased, we must rerun this again with dJ/2.
        3. Is dJ bigger or smaller than we've allowed it to be? If either is true, then adjust the allowed max/min norms (we may want to change this such that a global max/min is set instead of adjusting this global max and min according to another global max/min...we should only have a single set of bounds)
        2015-05-05

        Params:
        -------
        dJ
        dsisj
        **settings
        """
        keepsampling = True
        ntries = 0
        NTRIES_MAX = 5 # max number of times to retry metropolis sampling
        while keepsampling:
            self.generate_samples()
            self.runStatus['sampleSteps'] += 1
            ntries += 1
            self.sisj = self.calc_sisj()
            nextdsisj = self.constraints -self.sisj

            # Roll back updates if it ends up increasing error.
            if ( np.max(np.abs(nextdsisj))>(1.3*np.max(np.abs(dsisj))) ) and \
                    (ntries<NTRIES_MAX):
                dJ /= 2.
                self.J -= dJ # restore J to halfway between current and original value
                settings['max_dj_norm'] /= 2.
                self.runStatus['sampleSteps'] -= 1

                self.printout('Max error increased. Shrinking max_dj_norm to %f.' \
                        % settings['max_dj_norm'] + ' Sampling again...\n')
            # Case where error has been decreased.
            else:
                settings['max_dj_norm'] *= 2.
                keepsampling = False

            # Let's keep dJ norm between two reasonable floor and ceil values.
            if settings['max_dj_norm'] > settings['MAX_DJ_NORM']:
                settings['max_dj_norm'] = settings['MAX_DJ_NORM']
                self.printout('Lowering max_dj_norm to max value %f.\n' % settings['max_dj_norm'])
            elif settings['min_dj_norm'] < settings['MIN_DJ_NORM']:
                settings['min_dj_norm'] = settings['MIN_DJ_NORM']
                self.printout('Raising min_dj_norm to floor value %f.\n' % settings['min_dj_norm'])
        return dJ,nextdsisj
    
    def MCH(self,J0=None, 
                errmx=2e-2, errnormtol=None, samplesn=1e4, maxmchcount=10,
                maxlearnsteps=200, eta=1., etamch=.5, name='', alpha=0,
                mindjnorm=None,maxdjnorm=None,maxdjnormmch=None, 
                samplestepsmx=15,
                display=False):
        """
        Solve for the parameters Jij and hi in a pairwise maximum entropy model given the
        average pairwise correlations <sisj> and means <si> using just the MCH sampling algorithm. This is ideal for when we have a good approximation to the real solution and need to close the gap because this takes a long time to converge accurately (although theoretically it can converge exactly).

        maxdjnorm : the norm change allowed in J per iteration, max value at 1
            and below that is proportional to the error
        maxmchcount : max no. of consecutive iterations with MC histogram
            approx.
        alpha : weight given to inertial term when updating parameters
        errmx : max deviation allowed for any single fit to corresonding statistic
        errnormmx : tolerated error for norm of vector difference to statistics
        2015-05-01
        """
        if not J0 is None:
            self.J = J0

        # Solver settings.
        settings = {'MAX_LEARN_STEPS':maxlearnsteps, 'MAX_MCH_COUNT':maxmchcount, 
                    'ALPHA':alpha, 'ETA':eta, 'ETA_MCH':etamch, 
                    'MIN_DJ_NORM':1e-3,'MAX_DJ_NORM':np.sqrt(self.N),
                    'min_dj_norm':1e-3,'max_dj_norm':np.sqrt(self.N),
                    'MAX_DJ_NORM_MCH':self.N**.5,
                    'MCH_ERR_GROWTH_TOL':1.3, 'ERR_MX':errmx, 
                    'SAMPLE_STEPS_MX':samplestepsmx}

        # Initialize variables.
        allJ = np.zeros((maxlearnsteps+1,self.J.size))
        allsisj = np.zeros((maxlearnsteps+1,self.J.size))
        distNorm = [np.inf,]
        dJ = 0.
        sisjerrs = [] # keep track of vector of differences from estimated correlation to
                        # real correlations

        # Initialize counters and switches.
        self.runStatus = {'learnSteps':0,'sampleSteps':0,'mcHistCount':0}
        #learnSteps: total number of iterations of J that have happened  including MC hist and Metropolis sampling
        #samplesteps  number of times Metropolis sampling has happened
        #SIncremented boolean for whether S has been grown for error reduction

        # Get initial sample.
        self.generate_samples()
        self.sisj = self.calc_sisj()
        dsisj = self.constraints-self.sisj
        distNorm.append( np.linalg.norm(dsisj) )
        allJ[0,:], allsisj[0,:] = J0, self.sisj
        self.printout(np.max(np.abs(dsisj))) 
        
        # Solve for parameters J.
        while (np.max(np.abs(dsisj)) > settings['ERR_MX']) and \
                (self.runStatus['learnSteps'] <= settings['MAX_LEARN_STEPS']) and\
                (self.runStatus['sampleSteps']<= settings['SAMPLE_STEPS_MX']):
            sisjerrs.append(dsisj)
            dJ = dsisj*settings['ETA'] +settings['ALPHA']*dJ
            
            # Either MC hist normally if change dJ is not too large, but
            # otherwise only do for a few, set number of times samples
            if (np.linalg.norm(dJ) < settings['MAX_DJ_NORM']):
                learnStepsDelta,dsisj,dJ = self.iterate_MCH(dJ,distNorm,**settings)
            else:
                learnStepsDelta,dsisj,dJ = self.iterate_MCH(dJ,distNorm,**settings)
            self.runStatus['learnSteps'] += learnStepsDelta

            # Update variables.
            self.J += dJ
            prevdJ = dJ.copy()
            allJ[self.runStatus['sampleSteps']+1,:] = self.J
            prevdsisj = dsisj.copy()

            # If MC histogram has gotten us close enough, take it.
            # MCH cannot return fits to each individual statistic and can
            self.printout("Max deviation is %f. Max allowed is %f.\n" 
                          %(np.max(np.abs(dsisj)),settings['ERR_MX']) )
            if np.max(np.abs(dsisj)) < settings['ERR_MX']:
                break

            # Metropolis sample.
            # Metropolis sample typically increases norm error to statistics, but what we
            # wish to limit is only the maximum deviation for any given statistic.
            # If the max error grows too fast, roll back the step in dJ and recompute
            # with a smaller dJ, in this case with all entries halved.
            self.printout('\nSampling...\n')
            dJ,dsisj = self.sample_metropolis_dJ( dJ,dsisj,**settings )
            distNorm.append( np.linalg.norm(dsisj) )
            self.printout("Max deviation is %f\n" % np.max(np.abs(dsisj)) )
            
            # Output
            self.printout('learnSteps='+str(self.runStatus['learnSteps'])+'\n')
            # f.write('prevsisj='+str(prevsisj)+'\n')
            # f.write('J='+str(J)+'\n')
            self.printout('\tdist = '+str(distNorm[-1])+'\n')
            self.printout('dJNorm = '+str(np.linalg.norm(dJ))+'\n')
            self.printout('distNorm = '+str(distNorm)+'\n')

            allsisj[self.runStatus['sampleSteps'],:] = self.sisj
    #        if not (learnSteps%10):
    #        sio.savemat('temp_J',{'allJ_hat':allJ,'distNorm':distNorm,
    #                'allsisj':allsisj,'J':J})
            #sio.savemat('sisjerrs'+name,{'sisjerrs':sisjerrs})
            self.runStatus['learnSteps'] += 1

        # Exitflag steps.
        exitflag = 0
        if (np.max(np.abs(dsisj)) < settings['ERR_MX']) and \
                not (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 0
        elif not (np.max(np.abs(dsisj)) < settings['ERR_MX']) and \
                    (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 1
        elif (np.max(np.abs(dsisj)) < settings['ERR_MX']) and \
                (self.runStatus['learnSteps'] > settings['MAX_LEARN_STEPS']):
            exitflag = 2
        if self.runStatus['sampleSteps']>settings['SAMPLE_STEPS_MX']:
            exitflag = 3
            print("MC sampling steps exceed allowed max.")
        self.printout("Done")
        return exitflag

def get_jac_col(i,sisj,samples,eps,sym=False):
    """
    Return Jacobian estimate using MCH algorithm. The dJ used will is given as eps
    but this algorithm will adaptively adjust the interval depending on whether
    floating point precision will be good enough to estimate that interval.
    2015-05-01
    """
    dJ = np.zeros((sisj.size))
    dJ[i] = eps[i]
    sisjpos,Zfracisbad = sample_MCH(dJ,samples,sym=sym)
    sisjneg,Zfracisbad = sample_MCH(-dJ,samples,sym=sym)
    #while Zfracisbad:
    #    warnings.warn("Z frac is bad. Dynamically changing eps value.")
    #    eps /= 10.
    #    dJ[i] = eps
    #    sisj_,Zfracisbad = sample_MCH(dJ,samples,sym=sym)
    #    if eps<1e-6:
    #        raise Exception("eps very small.")
    #    print "eps = %1.3f" %eps
    return (sisjpos-sisjneg)/(2*eps)

def sample_MCH(dJ,samples,sym=False):
    """
    Sample Monte Carlo histogram.
    One problem here is that the calculated fraction of partition functions increases
    exponentially with the energy so it can easily become to large or too small for
    Python to deal with. In those cases, we automatically scale the given dJ in the
    right direction (positive or negative).

    See pages 13-14 in "Information in Justice and Conflict"
    2014-05-19
    """
    N = samples.shape[1]
    NN = N*(N-1)/2
    sisj = np.zeros((NN+N))
    ZFrac = 0.
    #ZFrac = np.float128(0.) # make sure we have high accuracy
    Zfracisbad = False

    # At limits of extrapolation, ZFrac will become very, very small.
    E = fast.calc_e(dJ,samples)
    expE = np.exp(-E)
    ZFrac = np.mean( expE )

    k = 0
    if not sym:
        for i in range(N):
            sisj[k] = np.mean( samples[:,i]*expE )
            k += 1
    for i in range(N-1):
        for j in range(i+1,N):
            sisj[k] = np.mean( samples[:,i]*samples[:,j]*expE )
            k +=1
    sisj /= ZFrac

    if ZFrac==0 or np.isnan(ZFrac):
        Zfracisbad = True

    return sisj,Zfracisbad

# ---------------------------------------------------------------------------------
class ExactNonlinear():
    """
    Front end to exact solver using tosolve functions in py_lib directory.
    2015-05-05
    """
    def __init__(self,*args):
        Model.__init__(self,*args)

    def solve(self,J0=None):
        from .exact import solve_ising
        if J0 is None:
            self.J = solve_ising(self.N,self.constraints,self.J,0,method='fast',maxfev=3e3)
        else:
            self.J = solve_ising(self.N,self.constraints,J0,0,method='fast',maxfev=3e3)

def main():
    return

def iterate_metropolis( nIters, sample, E, sample_metropolis ):
    """Helper function for joblib parallelization.
    2015-08-19
    """
    for j in range(nIters):
        de = sample_metropolis( sample[None,:],E )
        E += de
    return sample,E

if __name__=='__main__':
    main()
