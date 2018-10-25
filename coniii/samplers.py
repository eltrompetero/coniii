# ========================================================================================================= #
# Classes for sampling from Boltzmann type models.
# For HamiltonianMC (aka hybrid Monte Carlo and for only continuous state space systems), you should
# define functions calc_e() and grad_e() at the top of this file to use the jit speedup.
# 
# Eddie Lee edl56@cornell.edu
# 
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
# ========================================================================================================= #

#from numdifftools import Gradient
from numba import jit,njit,float64
from numpy import sin,cos,exp
from scipy.spatial.distance import squareform
import multiprocess as mp
from .utils import *
from datetime import datetime

# ------------------------------------------------------------------------------- #
# Define calc_e() and grad_e() functions here if you wish to use jit speedup!     #
# ------------------------------------------------------------------------------- #
# Some sample energy functions.
# 
##### Gaussian #####
# @jit(nopython=True)
# def calc_e(h,x):
#     return (h*x**2).sum()/2
# @jit(nopython=True)
# def grad_e(h,x):
#     return h*x
#
#
##### Heisenberg #####
# @jit(nopython=True)
#def calc_e(theta,x):
#    """
#    Heisenberg model. 
#    2016-08-16
#    
#    Parameters
#    ----------
#    theta (ndarray)
#        List of couplings Jij
#    x (ndarray)
#        List of angles (theta_0,phi_0,theta_1,phi_1,...,theta_n,phi_n)
#    """
#    n = len(x)//2  # number of spins
#    E = 0.
#    k = 0
#    for i in xrange(0,2*(n-1),2):
#        for j in xrange(i+2,2*n,2):
#            E += theta[k] * (sin(x[i])*sin(x[j])*
#                     (cos(x[i+1])*cos(x[j+1])+sin(x[i+1])*sin(x[j+1])) + 
#                     cos(x[i])*cos(x[j]))
#            k += 1
#    return E
#@jit(nopython=True)
#def grad_e(theta,x):
#    n = len(x)//2  # number of spins
#    g = np.zeros((len(x)))
#    
#    for i in xrange(0,2*n,2):
#        for j in xrange(0,2*n,2):
#            if i!=j:
#                k = sub_to_ind(n,i//2,j//2)
#                # Derivative wrt angle theta
#                g[i] += theta[k] * ( cos(x[i])*sin(x[j])*
#                                 (cos(x[i+1])*cos(x[j+1])+sin(x[i+1])*sin(x[j+1])) - 
#                                 sin(x[i])*cos(x[j]) )
#                # Derivative wrt angle phi
#                g[i+1] += theta[k] * (sin(x[i])*sin(x[j])*
#                                 (-sin(x[i+1])*cos(x[j+1])+cos(x[i+1])*sin(x[j+1])) )
#    return g
#
#
# ------------------------------------------------------------------------------- #
@jit(nopython=True)
def calc_e(theta, x):
    """
    Heisenberg model. 
    
    Parameters
    ----------
    theta : ndarray
        List of couplings Jij
    x : ndarray
        List of angles (theta_0,phi_0,theta_1,phi_1,...,theta_n,phi_n)
    """

    n = len(x)//2  # number of spins
    E = 0.
    k = 0
    for i in range(0,2*(n-1),2):
        for j in range(i+2,2*n,2):
            E += theta[k] * (sin(x[i])*sin(x[j])*
                     (cos(x[i+1])*cos(x[j+1])+sin(x[i+1])*sin(x[j+1])) + 
                     cos(x[i])*cos(x[j]))
            k += 1
    return -E

@jit(nopython=True)
def grad_e(theta, x):
    """
    Derivatives wrt the angles of the spins.
    """

    n = len(x)//2  # number of spins
    g = np.zeros((len(x)))
    
    for i in range(0,2*n,2):
        for j in range(0,2*n,2):
            if i!=j:
                k = sub_to_ind(n,i//2,j//2)
                # Derivative wrt angle theta_i
                g[i] += theta[k] * ( cos(x[i])*sin(x[j])*
                                 (cos(x[i+1])*cos(x[j+1])+sin(x[i+1])*sin(x[j+1])) - 
                                 sin(x[i])*cos(x[j]) )
                # Derivative wrt angle phi_i
                g[i+1] += theta[k] * (sin(x[i])*sin(x[j])*
                                 (-sin(x[i+1])*cos(x[j+1])+cos(x[i+1])*sin(x[j+1])) )
    return -g

@jit(nopython=True)
def grad_e_theta(theta, x):
    """
    Derivatives wrt the couplings theta. 
    """

    n = len(x)//2  # number of spins
    g = np.zeros((len(theta)))
    k = 0
    for i in range(0,2*(n-1),2):
        for j in range(i+2,2*n,2):
            g[k] = (sin(x[i])*sin(x[j])*(cos(x[i+1])*cos(x[j+1])+sin(x[i+1])*sin(x[j+1])) + 
                     cos(x[i])*cos(x[j]))
            k += 1
    return -g


class Sampler():
    """Base class for MCMC sampling."""
    def __init__(self, n, theta, **kwargs):
        """
        Parameters
        ----------
        n : int
            System size.
        theta : ndarray
            Lagrangian multipliers.
        """

        self.n = n
        self.theta = theta
        return

    def update_parameters(self, new_parameters):
        return
    
    def generate_samples(self, sample_size, **kwargs):
        """
        Parameters
        ----------
        sample_size : int
        """
        return

    def generate_samples_parallel(self, sample_size, **kwargs):
        """
        Parameters
        ----------
        sample_size : int
        """
        return
    
    def sample_metropolis(self, s, energy):
        """
        Parameters
        ----------
        s : ndarray
            State to perturb randomly.
        energy : float
            Energy of configuration.
        """
        return 
#end Sampler


# ============== #
# Wolff sampler. #
# ============== #
class WolffIsing(Sampler):
    def __init__(self, J, h):
        """
        Wolff cluster sampling for +/-1 Ising model.

        NOTE: This has not been properly tested.

        Parameters
        ----------
        J : ndarray
            Couplings as vector or matrix.
        h : ndarray
            Local fields.
        """

        assert len(J)==(len(h)*(len(h)-1)//2)

        from scipy.spatial.distance import squareform

        self.update_parameters(J,h) 
        self.n=len(h)
        self.rng=np.random.RandomState()
   
    def update_parameters(self, J, h):
        if J.ndim==1:
            J=squareform(J)
        self.J=J
        self.h=h

    def generate_sample(self, samplesize, n_iters, 
                        initialSample=None,
                        save_history=False,
                        ):
        """
        Generate samples by starting from random initial states.
        """

        if initialSample is None:
            sample = self.rng.choice([-1.,1.],size=(samplesize,self.n))
        else: sample = initialSample
        
        if save_history:
            history = np.zeros((samplesize,self.n,n_iters+1))
            history[:,:,0] = sample.copy()
            for i,s in enumerate(sample):
                for j in range(n_iters):
                    self.one_step(s,initialsite=j%self.n)
                    history[i,:,j+1] = s.copy()
            return sample,history

        else:
            for i,s in enumerate(sample):
                for j in range(n_iters):
                    self.one_step(s)
            return sample

    def generate_sample_parallel(self,samplesize,n_iters,
                                 initialSample=None):
        """
        Generate samples by starting from random or given initial states.
        """

        if initialSample is None:
            sample = self.rng.choice([-1.,1.],size=(samplesize,self.n))
        else: sample = initialSample
        #if (n_iters%self.n)!=0:
        #    n_iters = (n_iters//self.n+1)*self.n  # hit each spin the same number of times
        #    print "Increased n_iters to %d"%n_iters
        
        def f(args):
            s,seed=args
            self.rng=np.random.RandomState(seed)
            for j in range(n_iters):
                self.one_step(s,initialsite=j%self.n)
            return s
        
        pool=mp.Pool(mp.cpu_count())
        sample=np.vstack(pool.map(f,list(zip(sample,np.random.randint(2**31-1,size=samplesize)))))
        pool.close()
        return sample

    def _generate_sample(self,samplesize,sampleseparation):
        """
        Generate samples by evolving a single state.
        """

        n=self.n
        state=self.rng.randint(2,size=n)*2-1.
        sample=np.zeros((samplesize,n))
        
        nsamplesCounter=0
        counter=0
        while nsamplesCounter<samplesize:
            self.one_step(state)
            if (counter%sampleseparation)==0:
                sample[nsamplesCounter]=state.copy()
                nsamplesCounter+=1
            counter+=1
        return sample
            
    def one_step(self,state,initialsite=None):
        """
        Run one iteration of the Wolff algorithm that involves finding a cluster and possibly flipping it.
        """

        n,J,h=self.n,self.J,self.h
        self.expdJ = np.exp(-2*state[:,None]*J*state[None,:])

        # Choose random site.
        initialsite = initialsite or self.rng.randint(n)
        cluster = self.build_cluster(state,initialsite)
        
        # Flip cluster?
        # The first line is my addition to the algorithm that introduces a factor into the acceptance ratio
        # such that detailed balance is met and agrees with the ratio of the probabiltiies of the two possible
        # orientations of the cluster.
        if self.rng.rand()<np.exp(-2*h[cluster].dot(state[cluster])):
            state[cluster] *= -1 

    def build_cluster(self,state,initialsite):
        """
        Grow cluster from initial site.
        """

        n=self.n
        marked,newSites=[],[]
        newSites.append(initialsite)  # list of sites to continue exploring
        marked.append(initialsite)  # list of sites in the cluster
        
        # While there are new neighboring sites to explore from the ones already marked, keep going.
        while len(newSites)>0:
            newSites += self.find_neighbors(state,newSites[0],marked)
            newSites.pop(0)
        return marked
        
    def find_neighbors(self,state,site,alreadyMarked):
        """
        Return neighbors of given site that need to be visited excluding sites that have already been visited.
        This is the implementation of the Wolff algorithm for finding neighbors such that detailed balance is
        satisfied. I have modified to include random fields such tha the probability of adding a neighbors
        depends both on its coupling with the current site and the neighbor's magnetic field.

        Parameters
        ----------
        state
        site
        alreadyMarked
        """

        # Find neighbors in cluster.
        # If spins were parallel in original system, then the probability that a bond forms between is
        # (1-exp(-2*J))

        # Find neighbors but keep only neighbors that have not already been visited.
        ix = np.zeros((self.n),dtype=np.bool)
        ix[alreadyMarked] = True
        ix[site] = True
        neighbors = iterate_neighbors(self.n,ix,self.expdJ[:,site],self.rng.rand(self.n)).tolist()
        
        alreadyMarked += neighbors  # Add to list inplace.
        
        return neighbors

# Helper functions for WolffIsing.
@jit
def iterate_neighbors(n,ix,expdJ,r):
    """
    Iterate through all neighbors of a particular site and see if a bond should be formed between them.

    Parameters
    ----------
    n : int
        System size.
    ix : ndarray of bool
        Indices of sites that have already been visited.
    expdJ : ndarray
        np.exp( -2*state[:,None]*state[None,:]*J )
    r : ndarray
        Array of random numbers.
    """

    counter=0
    neighbors=np.zeros((n), dtype=int)
    for i in range(n):
        # Don't include neighbors that are already marked.
        # Check against probability as specified in Wolff's paper for forming link.
        # Must index r with i and not counter because otherwise we end up with cases where we are repeating
        # comparisons betweeen multiple i.
        if not ix[i] and r[i]<(1-expdJ[i]):
            neighbors[counter]=i
            counter += 1
    return neighbors[:counter]
#end WolffIsing



# ====================== #
# Swendsen-Wang sampler. #
# ====================== #
class SWIsing(Sampler):
    def __init__(self, n, theta, calc_e, nCpus=None, rng=None):
        """
        Swendsen-Wang sampler on Ising model with +/-1 formulation.

        NOTE: This has not been properly tested.

        Parameters
        ----------
        n : int
            Number of elements in system.
        theta : ndarray
            Vector of parameters in Hamiltonian.
        calc_e : function
            f( states, params )
        nCpus : int,0
            If None, then will use all available CPUs.
        rng : RandomState,None
        """
        raise NotImplementedError
        self.n = n
        self.theta = theta
        self.h,self.J = theta[:n],theta[n:]
        self.nCpus = nCpus or mp.cpu_count()
        
        self.calc_e = calc_e

        if rng is None:
            self.rng=np.random.RandomState()

    def generate_sample_parallel(self,n_samples,n_iters,
                                 initial_state=None,
                                 n_cpus=None):
        """
        Parameters
        ----------
        n_samples
        n_iters
        initial_state : ndarray,None
        """

        n_cpus = n_ncpus or self.nCpus

        if initial_state is None:
            sample = self.rng.choice([-1,1],size=(n_samples,self.n))
        else:
            sample = initial_state

        def f(params):
            i,seed,state = params
            self.rng = np.random.RandomState(seed)
            for j in range(n_iters):
                self.one_step(state)
            return state

        p = mp.Pool(n_cpus)
        self.sample = np.vstack( p.map(f,list(zip(list(range(n_samples)),
                                             self.rng.randint(2**31,size=n_samples),
                                             sample))) )
        p.close()

    def generate_sample(self, n_samples, n_iters, initial_state=None):
        """
        Parameters
        ----------
        n_samples
        n_iters
        initial_state : ndarray,None
        """

        if initial_state is None:
            sample = self.rng.choice([-1,1],size=(n_samples,self.n))
        else:
            sample = initial_state

        for i in range(n_samples):
            for j in range(n_iters):
                self.one_step(sample[i])

        self.sample = sample

    def one_step(self,state):
        self._clusters = self.get_clusters(state)
        self.randomly_flip_clusters(state,self._clusters)
        return state
    
    def get_clusters(self, state):
        """
        Get a random sample of clusters.
        """

        n,J = self.n,self.J

        # Generate random edges between neighbors.
        p = 1-np.exp(-2*self.J*pairwise_prod(state))
        adj = squareform( sample_bonds(p,self.rng.rand(len(J)),state,J) )
        
        return iter_cluster(adj)

    def randomly_flip_clusters(self, state, clusters):
        for cls in clusters:
            if self.rng.rand()>(1/(1+np.exp(-2*self.h[cls].dot(state[cls])))):
                state[cls] *= -1 

    def print_cluster_size(self, n_iters):
        n_samples = 1
        sample = self.rng.choice([-1,1],size=(n_samples,self.n))

        for j in range(n_iters):
            self.one_step(sample[0])
            print(np.mean([len(c) for c in self._clusters]))

@jit(nopython=True)
def pairwise_prod(state):
    counter = 0
    n = len(state)
    prod = np.zeros((n*(n-1)//2))
    for i in range(n-1):
        for j in range(i+1,n):
            prod[counter] = state[i]*state[j]
            counter += 1
    return prod
        
@jit(nopython=True)
def sample_bonds(p, r, state, J):
    """
    Parameters
    ----------
    p : ndarray
        Probability of bond formation.
    r : ndarray
        Random numbers.
    state
    J
    """

    n = len(state)
    bonds = np.zeros((len(J)))
    counter=0
    for i in range(n-1):
        for j in range(i+1,n):
            if J[counter]<0 and state[i]*state[j]<0 and r[counter]<p[counter]:
                bonds[counter] = 1
            elif J[counter]>0 and state[i]*state[j]>0 and r[counter]<p[counter]:
                bonds[counter] = 1
            counter += 1
    return bonds

def iter_cluster(adj):
    """
    Cycle through all spins to get clusters.
    """

    initialSites = set(range(len(adj)))
    marked = set()  # all sites that have been clustered
    clusters = []

    while initialSites:
        thisCluster = []
        newSites = [initialSites.pop()]  # list of sites to continue exploring
        marked.add(newSites[0])

        # While there are new neighboring sites to explore from the ones already marked, keep going.
        while len(newSites)>0:
            thisCluster.append(newSites[0])
            
            neighbors = set(np.nonzero(adj[newSites[0]])[0].tolist())
            neighbors.difference_update(marked)
            
            newSites += neighbors
            marked.update(neighbors)
            
            newSite0 = newSites.pop(0)
        
        clusters.append(thisCluster)
        initialSites.difference_update(marked)
    return clusters

def spec_cluster(L,exact=True):
    """
    Parameters
    ----------
    L : ndarray
        Graph Laplacian
    """

    from scipy.linalg import eig

    eig,v = eig(L)
    if exact:
        clusterix = np.nonzero(eig==0)[0]
    else:
        clusterix = np.nonzero(np.isclose(eig,0))[0]
    cluster = []
    for i in clusterix:
        cluster.append(np.nonzero(np.isclose(v[:,i],0)==0)[0])
    return cluster
#end SWIsing



# =========================== #
# Parallel tempering sampler. #
# =========================== #
class ParallelTempering(Sampler):
    def __init__(self,n,theta,calc_e,temps,
                 sample_size=1000,
                 replica_burnin=100,
                 rng=None):
        """
        Run multiple replicas in parallel at different temperatures using Metropolis sampling to
        equilibrate.

        NOTE: This has not been properly tested.

        Parameters
        -------------
        n : int
        theta : ndarray
            Mean field and coupling parameters.
        temps : list-like
        rng : numpy.RandomState,None
        """

        raise NotImplementedError
        assert len(temps)>=2

        self.n = n
        self.theta = [theta/T for T in temps]
        self.calc_e = calc_e
        self.temps = temps
        self.sampleSize = sample_size
        self.sample = None
        self.replicaBurnin = replica_burnin
        self.rng = rng or np.random.RandomState()
        
        self.setup_replicas(replica_burnin)
    
    def update_parameters(self, theta=None):
        """
        Update parameters for each replica.
        """

        if theta is None:
            theta = self.theta[0]

        self.theta = [theta/T for T in self.temps]
        for theta,rep in zip(self.theta,self.replicas):
            rep.theta = theta
            rep.h,rep.J = theta[:self.n],squareform(theta[self.n:])

    def setup_replicas(self, burnin):
        """
        Initialize a set of replicas at different temperatures using the Metropolis algorithm as coded in
        FastMCIsing.
        """

        self.nReps = len(self.temps)

        self.replicas = []
        for theta,T in zip(self.theta,self.temps):
            self.replicas.append( FastMCIsing(self.n,theta,self.calc_e) )
            # Burn replica in.
            self.replicas[-1].generate_samples_parallel(self.sampleSize,n_iters=burnin)
        self.sample = self.replicas[0].samples
        
    def one_step(self, pool, burn_factor, exchange=True):
        """
        Parameters
        ----------
        pool : mp.multiprocess.Pool
        burn_factor : int
            Number of times to iterate through the system.
        exchange : bool,True
            Run exchange sampling. This is typically turned off when you just want to burn in the replicas.
        """

        reps = self.replicas
        
        if exchange:
            E = np.zeros((self.sampleSize,len(self.temps)))
            for i,r in enumerate(reps):
                E[:,i] = r.E[:].ravel()*self.temps[i]
            
            # Iterate through each pair of systems by adjacent temperature starting from the lowest T.
            exchangeix = np.zeros((self.sampleSize,len(self.temps)-1),dtype=np.bool)
            for i in range(1,len(self.temps)):
                exchangeix[:,i-1] = ( self.rng.rand(self.sampleSize) <
                            np.exp((E[:,i]-E[:,i-1])*(1/self.temps[i]-1/self.temps[i-1])) )
                if exchangeix[:,i-1].any():
                    # Exchange samples and exchange energies.
                    tempSample = reps[i-1].samples[exchangeix[:,i-1]]
                    reps[i-1].samples[exchangeix[:,i-1]] = reps[i].samples[exchangeix[:,i-1]]
                    reps[i].samples[exchangeix[:,i-1]] = tempSample
                    
                    tempE = E[exchangeix[:,i-1],i-1]
                    E[exchangeix[:,i-1],i-1] = E[exchangeix[:,i-1],i]
                    E[exchangeix[:,i-1],i] = tempE
                else:
                    print("No overlap between replica %d and %d"%(i,i-1))

        # Evolve replicas by self.n*burn_factor iterations.
        def f(r):
            r.generate_samples( self.sampleSize,
                                initialSample=r.samples,
                                n_iters=self.n*burn_factor,
                                systematic_iter=True )
            return r
        self.replicas = pool.map(f,reps)

        if exchange:
            return exchangeix

    def generate_samples(self,n_iters=100,
                         initial_burn_factor=10,
                         final_burn_factor=10,
                         burn_factor=1,
                         save_sample=True):
        """
        Burn in, run replica exchange simulation, then burnin.

        Parameters
        ----------
        n_iters : int,100
            Number of times to run the RMC. This involves sampling from each replica N*burn_factor  times.
        initial_burn_factor,final_burn_factor (int=10)
            Number of time to iterate through system to burn in at the beginning and at the end.
        burn_factor : int,1
            Passed into one_step().
        """
        if initial_burn_factor>0:
            self.burn_in(initial_burn_factor)
        
        pool = mp.Pool(mp.cpu_count())
        for i in range(1,n_iters):
            self.one_step(pool,burn_factor)
        pool.close()
        
        if final_burn_factor>0:
            self.burn_in(final_burn_factor)
        if save_sample:
            self.sample = self.replicas[0].samples

    def generate_trajectory(self,n_iters=10,burn_factor=5):
        """
        Run MC and save at each iteration. Note that this will overwrite current samples stored in replicas.
        Save a sample every exchange and a burn factor step.

        Parameters
        ----------
        n_iters : int,10
        burn_factor : int,5

        Returns
        -------
        repSamples : list of ndarrays
        """

        repSamples = [np.zeros((self.sampleSize,self.n,n_iters)) for i in range(self.nReps)]
        pool = mp.Pool(mp.cpu_count())
        for i in range(n_iters):
            self.one_step(pool,burn_factor)
            for r,rep in zip(repSamples,self.replicas):
                r[:,:,i] = rep.samples[:,:]
        pool.close()
        return repSamples

    def autocorr_spin(self,n_iters,burn_factor):
        """
        Calculate spin autocorrelation.
        E[s(t)s(t+dt)]

        Parameters
        ----------
        n_iters
        burn_factor

        Returns
        -------
        autocorr
        repSamples
        """

        from misc.stats import acf
        autocorr = np.zeros((self.nReps,n_iters//2))
        repSamples = self.generate_trajectory(n_iters,burn_factor)
        for i,r in enumerate(repSamples):
            r = np.rollaxis(r,1).reshape(self.n*self.sampleSize,n_iters)
            # This will be problematic (and return nans) if the state doesn't change at all from Metropolis
            # updates.
            autocorr[i,:] = np.nanmean( acf(r,axis=1),axis=0 )
        return autocorr,repSamples

    def pn(self,n_iters=1,initial_burn_factor=10,burn_factor=1):
        """
        Estimate acceptance probabilities of exchange between two adjacent temperatures. The acceptance
        probability is ordered from exchange between replicas i and i+1 starting from i=0.

        Estimate stay time at any particular replica.

        Parameters
        ----------
        n_iters : int,100
            Number of times to run the RMC. This involves sampling from each replica N*burn_factor times.
        initial_burn_factor : int,10
            Burn factor right after setting up random state.
        burn_factor : int,1
            Passed into one_step().
        """ 

        # See how many samples are exchanged.
        # Save current state of sample while we estimate the acceptance probabilities.
        oldSample = self.sample.copy()
        
        # Sample for calculating exchange probabilities.
        pool = mp.Pool(mp.cpu_count())
        self.one_step(pool,initial_burn_factor,exchange=False)
        for i in range(n_iters):
            if i==0:
                exchangeix = self.one_step(pool,burn_factor)
            else:
                exchangeix = np.vstack((exchangeix,self.one_step(pool,burn_factor)))
        pool.close()
        
        # Place sample back to original state and save current set of samples.
        self._samples = [rep.samples for rep in self.replicas]
        self.sample = oldSample
        
        return exchangeix.mean(0),exchangeix

    def tn(self, exchangeix):
        """
        Estimate stay time at any particular replica given number of exchanges that happened during ReMC steps.
        Find stay time at replicas. This is the inverse of the probability that you switch out of a
        particular replica. Remember that replicas on the boundaries can only exchange with one other
        replica.

        Parameters
        ----------
        exchangeix : ndarray
        """ 

        nReps = len(self.temps)
        stayTime = np.zeros((nReps))
        for i in range(nReps):
            if i==0:
                stayTime[0] = 1/exchangeix[:,0].mean()
            elif i==(nReps-1):
                stayTime[-1] = 1/exchangeix[:,-1].mean()
            else:
                stayTime[i] = 1/exchangeix[:,i-1:i].sum(1).mean()

        return stayTime
    
    def exchange_measures(self, pn_kwargs={}):
        """
        Wrapper for computing exchange probabilities and stay times.
        """

        pn,exchangeix = self.pn(**pn_kwargs)
        tn = self.tn(exchangeix)
        return pn,tn

    def iterate_beta(self, tau):
        """
        Parameters
        ----------
        tau : ndarray
            Effective time at each temperature.
        """

        beta = 1/np.array(self.temps)
        tauEff = tau.copy()
        tauEff[0] /= 2
        tauEff[-1] /=2
        a = (beta[1:]-beta[:-1])/(tauEff[1:]+tauEff[:-1])
        c = a.sum()

        for i in range(1,self.nReps-1):
            beta[i] = beta[i-1] + a[i-1]*(beta[-1]-beta[0])/c
        return beta

    def optimize(self,
                 pn_kwargs={'n_iters':5},
                 max_iter=10,
                 disp=False,
                 save_history=False):
        """
        Apply algorithm from Kerler and Rehberg (1994) for finding fixed point for optimal temperatures.
        Optimized temperatures rewrite self.temps.

        Parameters
        ----------
        pn_kwargs : dict,{'n_iters':5}
        max_iter : int,10
            Number of times to iterate algorithm for beta. Each iteration involves sampling from REMC.
        disp : bool,False
            Print out updated parameters.
        save_history : bool,False
            If true, return history.
        """

        print("Optimizing REMC parameters...")

        for i in range(max_iter):
            pn,tn = self.exchange_measures(pn_kwargs=pn_kwargs)
            if disp:
                print("Iteration %d"%i)
                print("P(n): probability of exchange")
                print(pn)
                print("T(n): replica steps spent at temperature")
                print(tn)
            beta = self.iterate_beta(tn)
            
            self.temps = 1/beta
            self.update_parameters()
            self._pn = pn
            self._tn = tn

        print("After optimization:")
        print("Temperatures:")
        print(self.temps)
        print("Exchange probability:")
        print(pn)
        print("Suggested replica steps: %d"%(2/np.prod(pn)))  # Calculated from time to diffuse.
        print("Persistence time:")
        print(tn)

    def burn_in(self, n_iter):
        """
        Wrapper for iterating sampling without exchanging replicas.

        Parameters
        ----------
        n_iters : int
            Number of times to iterate through system.
        """

        pool = mp.Pool(mp.cpu_count())
        self.one_step(pool,n_iter,exchange=False)
        self.sample = self.replicas[0].samples
        pool.close()
#end ParallelTempering


# ============================ #
# Simulated tempering sampler. #
# ============================ #
class SimulatedTempering(Sampler):
    def __init__(self, n, theta, calc_e, temps, 
                 sample_size=1000,
                 replica_burnin=100,
                 rng=None,
                 method='single'):
        """
        NOTE: This has not be properly tested.

        Parameters
        ----------
        n : int
        theta : ndarray
        temps : list-like
        rng : numpy.RandomState,None
        method : str,'single'
            Choose between 'single' and 'multiple'. In the former, a single state is simulated while
            changing temperatures and in the latter a set of replicas at multiple temperatures are
            evolved simultaneously.
        """

        raise NotImplementedError("This is just a copy of the old Replica MC with some code for calculating the weighting function g(beta) that I didn't want to delete.")
        self.n = n
        self.theta = [theta/T for T in temps]
        self.calc_e = calc_e
        self.temps = temps
        self.sampleSize = sample_size
        self.sample = None
        self.replicaBurnin = replica_burnin
        self.rng = rng or np.random.RandomState()
        self.sampler = FastMCIsing(n,theta,calc_e)
        self.sampler.samples = np.zeros((1,n))
        self.gn = np.zeros((len(temps)))
        
    def update_parameters(self,theta=None):
        if theta is None:
            theta = self.theta[0]

        self.theta = [theta/T for T in self.temps]
        for theta,rep in zip(self.theta,self.replicas):
            rep.theta = theta
            rep.h,rep.J = theta[:self.n],squareform(theta[self.n:])

    def one_loop(self,sample,burn_factor,rng=None):
        """
        Metropolis sample state til it reaches the highest temperature and returns to T=1. At each new
        temperature, the sample is iterated burn_factor times.
        2017-03-17

        Parameters
        ----------
        pool (mp.multiprocess.Pool)
        burn_factor (int)
            Number of times to iterate through the system.
        exchange (bool=True)
            Run exchange sampling. This is typically turned off when you just want to burn in the replicas.
        """
        tempix = 0
        beta = [1/t for t in self.temps]
        self.sampler.samples[:] = sample
        self.sampler.rng=rng
        E = self.calc_e(sample,self.theta[0])
        
        reachedEnd = False
        while not reachedEnd:
            # Evolve temperature.
            if self.rng.rand()<np.exp(-E*(beta[tempix]-beta[tempix+1])+self.gn[tempix]-self.gn[tempix+1]):
                tempix += 1
            self.sampler.generate_samples(1,n_iters=burn_factor*self.n)
            E = self.calc_e(self.sampler.samples,self.theta[0])
            
            if tempix==(len(beta)-1):
                reachedEnd = True

        loopedAround = False
        while not loopedAround:
            # Evolve temperature.
            if self.rng.rand()<np.exp(-E*(beta[tempix]-beta[tempix-1])+self.gn[tempix]-self.gn[tempix-1]):
                tempix -= 1
            self.sampler.generate_samples(1,n_iters=burn_factor*self.n)

            if tempix==0:
                loopedAround = True
        sample = self.sampler.samples
        return sample

    def generate_samples(self,n_iters=100,
                         initial_burn_factor=10,
                         final_burn_factor=10,
                         burn_factor=1):
        """
        Run replica exchange simulation then burnin.
        2017-03-01

        Parameters
        ----------
        n_iters (int=100)
            Number of times to run the RMC. This involves sampling from each replica N*burn_factor  times.
        initial_burn_factor,final_burn_factor (int=10)
            Number of time to iterate through system to burn in at the beginning and at the end.
        burn_factor (int=1)
            Passed into one_step().
        """
        #self.burn_in(initial_burn_factor)
        
        def f(sample):
            rng = np.random.RandomState()
            for i in range(n_iters):
                sample = self.one_loop(sample,burn_factor,rng)
            return sample

        pool = mp.Pool(mp.cpu_count())
        self.sample = np.vstack( pool.map(f,self.rng.choice([-1,1],size=(self.sampleSize,1,self.n))) )
        pool.close()
        
        #self.burn_in(final_burn_factor)
        #self.sample = self.replicas[0].samples

    def pn(self,n_iters=1,initial_burn_factor=10,burn_factor=1):
        """
        Estimate acceptance probabilities of exchange between two adjacent temperatures. The acceptance
        probability is ordered from exchange between replicas i and i+1 starting from i=0.

        Estimate stay time at any particular replica.
        2017-03-01

        Parameters
        ----------
        n_iters (int=100)
            Number of times to run the RMC. This involves sampling from each replica N*burn_factor times.
        initial_burn_factor (int=10)
            Burn factor right after setting up random state.
        burn_factor (int=1)
            Passed into one_step().
        """ 
        # See how many samples are exchanged.
        # Save current state of sample while we estimate the acceptance probabilities.
        oldSample = self.sample.copy()
        
        # Sample for calculating exchange probabilities.
        pool = mp.Pool(mp.cpu_count())
        self.one_step(pool,initial_burn_factor,exchange=False)
        for i in range(n_iters):
            if i==0:
                exchangeix = self.one_step(pool,burn_factor)
            else:
                exchangeix = np.vstack((exchangeix,self.one_step(pool,burn_factor)))
        pool.close()
        
        # Place sample back to original state and save current set of samples.
        self._samples = [rep.samples for rep in self.replicas]
        self.sample = oldSample
        
        return exchangeix.mean(0),exchangeix

    def tn(self,exchangeix):
        """
        Estimate stay time at any particular replica given number of exchanges that happened during ReMC steps.
        Find stay time at replicas. This is the inverse of the probability that you switch out of a
        particular replica. Remember that replicas on the boundaries can only exchange with one other
        replica.
        2017-03-01

        Parameters
        ----------
        exchangeix (ndarray)
        """ 
        nReps = len(self.temps)
        stayTime = np.zeros((nReps))
        for i in range(nReps):
            if i==0:
                stayTime[0] = 1/exchangeix[:,0].mean()
            elif i==(nReps-1):
                stayTime[-1] = 1/exchangeix[:,-1].mean()
            else:
                stayTime[i] = 1/exchangeix[:,i-1:i].sum(1).mean()

        return stayTime
    
    def exchange_measures(self,pn_kwargs={}):
        """
        Wrapper for computing exchange probabilities and stay times.
        """
        pn,exchangeix = self.pn(**pn_kwargs)
        tn = self.tn(exchangeix)
        return pn,tn

    def reweighted_gn(self,listOfSample):
        """
        2017-03-01
        """
        from scipy.special import logsumexp
        E = [self.calc_e(s,theta) for s,theta in zip(listOfSample,self.theta)] # assuming that first 
                                                                              # temperature is 1
        E = np.vstack(E).T
        gn = -logsumexp( -E,axis=0 )
        
        # Extrapolate what gn should be by using samples from neighboring replicas. Since samples in the
        # middle have two neighbors, we take the mean of the adjacent extrapolations.
        # The boundpoints will take the average of their current value and the updated value from the
        # neighbor, effectively convergence with inertia.
        extrapgn = np.zeros((self.nReps))
        extrapgn[:-1] += -logsumexp( -E[:,1:]*self.temps[1:]/self.temps[:-1],axis=0 )
        extrapgn[1:] += -logsumexp( -E[:,:-1]*self.temps[:-1]/self.temps[1:],axis=0 )
        extrapgn[-1] += gn[-1]
        extrapgn[0] += gn[0]
        return extrapgn/2

    def iterate_beta(self,tau):
        """
        2017-03-01
        Parameters
        ----------
        tau (ndarray)
            Effective time at each temperature.
        """
        beta = 1/np.array(self.temps)
        tauEff = tau.copy()
        tauEff[0] /= 2
        tauEff[-1] /=2
        a = (beta[1:]-beta[:-1])/(tauEff[1:]+tauEff[:-1])
        c = a.sum()

        for i in range(1,self.nReps-1):
            beta[i] = beta[i-1] + a[i-1]*(beta[-1]-beta[0])/c
        return beta

    def optimize(self,
                 interp_kwargs={'kind':'quadratic'},
                 pn_kwargs={'n_iters':5},
                 max_iter=10,
                 disp=False,
                 save_history=False,
                 threshold=1e-2):
        """
        Apply algorithm from Kerler and Rehberg (1994) for finding fixed point for optimal temperatures.
        Optimized temperatures rewrite self.temps.
        2017-03-01

        Parameters
        ----------
        interp_kwargs (dict={'kind':'quadratic'})
            Interpoloation for gn(bn) = bn to update gn after beta update step.
        pn_kwargs (dict={'n_iters':5})
        max_iter (int=10)
            Number of times to iterate algorithm for beta. Each iteration involves sampling from REMC.
        disp (bool=False)
            Print out updated parameters.
        save_history (bool=False)
            If true, return history.
        threshold (float=1e-2)
        """
        print("Optimizing REMC parameters...")
        from scipy.interpolate import interp1d

        for i in range(max_iter):
            pn,tn = self.exchange_measures(pn_kwargs=pn_kwargs)
            if disp:
                print(self.gn,pn,tn)
            gn = self.reweighted_gn(self._samples)
            gn_of_beta = interp1d( 1/np.array(self.temps), gn, **interp_kwargs )
            beta = self.iterate_beta(tn)
            gnprime = gn_of_beta(beta)
            gnprime -= gnprime.min()
            
            self.temps = 1/beta
            self.update_parameters()
            self.gn = gnprime
            self._pn = pn
            self._tn = tn

        print("After optimization:")
        print("Temperatures:")
        print(self.temps)
        print("Exchange probability:")
        print(pn)
        print("Suggested replica steps: %d"%(2/np.prod(pn)))
        print("Persistence time:")
        print(tn)

    def burn_in(self,n_iter):
        """
        Wrapper for iterating sampling without exchanging replicas.
        2017-02-27

        Parameters
        ----------
        n_iters (int)
            Number of times to iterate through system.
        """
        pool = mp.Pool(mp.cpu_count())
        self.one_step(pool,n_iter,exchange=False)
        self.sample = self.replicas[0].samples
        pool.close()
#end SimulatedTempering



# =================== #
# Metropolis sampler. #
# =================== #
class FastMCIsing(Sampler):
    def __init__(self, n, theta,
                 n_cpus=None,
                 rng=None,
                 use_numba=True):
        """
        MC sample on Ising model with +/-1 formulation. Fast metropolis sampling by assuming form of
        Hamiltonian is an Ising model.

        Parameters
        ----------
        n : int
            Number of elements in system.
        theta : ndarray
            Concatenated vector of the field hi and couplings Jij. They should be ordered in
            ascending order 0<=i<n and for Jij in order of 0<=i<j<n.
        n_cpus : int,0
            If None, then will use all available CPUs minus 1.
        rng : RandomState,None
        use_numba : bool,True
            If True, use jit to speed up sampling, but random seed generator cannot be set by the user if this
            is switched on. If having problems with numba, this can be switched off.
        """
        
        self.n = n
        self.update_parameters(theta)
        self.nCpus = n_cpus or mp.cpu_count()-1
        
        self.calc_e, calc_observables, mchApproximation = define_ising_helper_functions()
        
        if use_numba:
            rng=None
        else:
            if rng is None:
                self.rng = np.random.RandomState()
        self.setup_sampling(use_numba)

    def update_parameters(self, theta):
        self.theta = theta
        self.h, self.J = theta[:self.n], squareform(theta[self.n:])

    def setup_sampling(self, use_numba):
        if use_numba:
            self.sample_metropolis = self._jit_sample_metropolis
            self.generate_samples = self._jit_generate_samples
            self.generate_samples_parallel = self._jit_generate_samples_parallel
        else:
            self.sample_metropolis = self._sample_metropolis
            self.generate_samples = self._generate_samples
            self.generate_samples_parallel = self._generate_samples_parallel

    def _jit_generate_samples(self,
                         sample_size,
                         n_iters=1000,
                         saveHistory=False,
                         initial_sample=None,
                         systematic_iter=False):
        """
        Generate Metropolis samples using a for loop and save samples in self.samples.

        Parameters
        ----------
        sample_size : int
            Number of samples to take.
        n_iters : int,1000
            Number of iterations to run Metropolis sampler.
        saveHistory : bool,False
            If True, the energy of the system after each Metropolis sampling step will be recorded.
            This can be useful for making sure that the system has reached equilibrium.
        initial_sample : ndarray,None
            If this is given, then this is used as the starting point for the Markov chain instead
            of a random sample.
        systematic_iter : bool,False
            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
            receive in equal number of chances to flip.
        """
        
        sample_metropolis = self._jit_sample_metropolis

        if initial_sample is None:
            samples = np.random.choice([-1.,1.],size=(sample_size,self.n))
        else:
            samples = initial_sample
        E = self.calc_e( samples, self.theta )
        h, J = self.h, self.J
        n = self.n
        
        if saveHistory:
            if systematic_iter:
                @jit('float64[:,:]()', locals={'samples':float64[:,:]}, forceobj=True)
                def sample():
                    history=np.zeros((sample_size,n_iters+1))
                    history[:,0]=E.ravel()
                    for i in range(sample_size):
                        for j in range(n_iters):
                            de = sample_metropolis( samples[i], h, J, j%n )
                            E[i] += de
                            history[i,j+1]=E[i]
                    return history
            else:
                @jit('float64[:,:]()', locals={'samples':float64[:,:]}, forceobj=True)
                def sample():
                    history = np.zeros((sample_size,n_iters+1))
                    history[:,0] = E.ravel()
                    for i in range(sample_size):
                        for j in range(n_iters):
                            de = sample_metropolis( samples[i], h, J, np.random.randint(n) )
                            E[i] += de
                            history[i,j+1] = E[i]
                    return history

            history = sample()
            self.E = E
            self.samples = samples
            return history
        else:
            if systematic_iter:
                def sample():
                    for i in range(sample_size):
                        for j in range(n_iters):
                            de = sample_metropolis( samples[i], h, J, j%n )
                            E[i] += de
            else:
                def sample():
                    for i in range(sample_size):
                        for j in range(n_iters):
                            de = sample_metropolis( samples[i], h, J, np.random.randint(n) )
                            E[i] += de

        sample()
        # read out variables
        self.E = E
        self.samples = samples

    def _generate_samples(self,
                         sampleSize,
                         n_iters=1000,
                         saveHistory=False,
                         initialSample=None,
                         systematic_iter=False):
        """
        Generate Metropolis samples using a for loop and save samples in self.samples.

        Parameters
        ----------
        sampleSize : int
            Number of samples to take.
        n_iters : int,1000
            Number of iterations to run Metropolis sampler.
        saveHistory : bool,False
            If True, the energy of the system after each Metropolis sampling step will be recorded.
            This can be useful for making sure that the system has reached equilibrium.
        initialSample : ndarray,None
            If this is given, then this is used as the starting point for the Markov chain instead
            of a random sample.
        systematic_iter : bool,False
            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
            receive in equal number of chances to flip.
        """

        if initialSample is None:
            self.samples = self.rng.choice([-1.,1.],size=(sampleSize,self.n))
        else: self.samples = initialSample
        self.E = self.calc_e( self.samples, self.theta )
        
        if saveHistory:
            if systematic_iter:
                history=np.zeros((sampleSize,n_iters+1))
                history[:,0]=self.E.ravel()
                for i in range(sampleSize):
                    for j in range(n_iters):
                        de = self._sample_metropolis( self.samples[i], flip_site=j%self.n )
                        self.E[i] += de
                        history[i,j+1]=self.E[i]
                return history
            else:
                history=np.zeros((sampleSize,n_iters+1))
                history[:,0]=self.E.ravel()
                for i in range(sampleSize):
                    for j in range(n_iters):
                        de = self._sample_metropolis( self.samples[i] )
                        self.E[i] += de
                        history[i,j+1]=self.E[i]
                return history
        else:
            if systematic_iter:
                for i in range(sampleSize):
                    for j in range(n_iters):
                        de = self._sample_metropolis( self.samples[i], flip_site=j%self.n )
                        self.E[i] += de
            else:
                for i in range(sampleSize):
                    for j in range(n_iters):
                        de = self._sample_metropolis( self.samples[i], self.rng, self.h, self.J )
                        self.E[i] += de
    
    def _jit_generate_samples_parallel(self,
                                       sample_size,
                                       n_iters=1000,
                                       cpucount=None,
                                       initial_sample=None,
                                       systematic_iter=False):
        """
        Metropolis sample multiple states in parallel and save them into self.samples.

        Parameters
        ----------
        sample_size : int
            Number of samples to take.
        n_iters : int,1000
            Number of iterations to run Metropolis sampler.
        initial_sample : ndarray,None
            If this is given, then this is used as the starting point for the Markov chain instead
            of a random sample.
        systematic_iter : bool,False
            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
            receive in equal number of chances to flip.
        """

        cpucount=cpucount or self.nCpus
        sample_metropolis = self._jit_sample_metropolis
        h, J = self.h, self.J
        n = self.n
        
        # setup case where initial_sample is not given
        if initial_sample is None:
            calc_e = self.calc_e
            theta = self.theta

            if not systematic_iter:
                @njit
                def f(args):
                    seed, theta = args
                    np.random.seed(seed)
                    s = np.zeros((1,n))

                    for i in range(n):
                        s[0,i]=np.random.randint(2)*2-1
                    E = calc_e(s, theta)

                    for j in range(n_iters):
                        de = sample_metropolis( s[0], h, J, np.random.randint(n) )
                        E += de
                    return s, E
            else:
                @njit
                def f(args):
                    seed, theta = args
                    np.random.seed(seed)
                    s = np.zeros((1,n))

                    for i in range(n):
                        s[0,i] = np.random.randint(2)*2-1
                    E = calc_e(s, theta)

                    for j in range(n_iters):
                        de = sample_metropolis( s[0], h, J, j%n )
                        E += de
                    return s, E
            
            args=((i,theta) for i in np.random.randint(2**31-1, size=sample_size))

        # setup case where initial_sample is given
        else:
            assert len(initial_sample)==sample_size, "Given sample is of wrong length."
            assert initial_sample.shape[1]==self.n, "Given sample is wrong dimension."
            self.E = self.calc_e( initial_sample, self.theta )

            if not systematic_iter:
                @njit
                def f(args):
                    s, E, seed = args
                    np.random.seed(seed)
                    for j in range(n_iters):
                        de = sample_metropolis( s, h, J, np.random.randint(n) )
                        E += de
                    return s, E
            else:
                @njit
                def f(args):
                    s, E, seed = args
                    np.random.seed(seed)
                    for j in range(n_iters):
                        de = sample_metropolis( s, h, J, j%n )
                        E += de
                    return s, E
            args = zip(initial_sample, self.E, np.random.randint(2**31-1, size=sample_size))
        
        # run sampling
        pool = mp.Pool(cpucount)
        self.samples, self.E = list(zip(*pool.map(f, args)))
        pool.close()
        
        # save results of sampling into instance data members
        self.samples = np.vstack(self.samples)
        self.E = np.array(self.E)

    def _generate_samples_parallel(self,
                                   sampleSize,
                                   n_iters=1000,
                                   cpucount=None,
                                   initial_sample=None,
                                   systematic_iter=False):
        """
        Metropolis sample multiple states in parallel and save them into self.samples.

        Parameters
        ----------
        sampleSize : int
            Number of samples to take.
        n_iters : int,1000
            Number of iterations to run Metropolis sampler.
        saveHistory : bool,False
            If True, the energy of the system after each Metropolis sampling step will be recorded.
            This can be useful for making sure that the system has reached equilibrium.
        initialSample : ndarray,None
            If this is given, then this is used as the starting point for the Markov chain instead
            of a random sample.
        systematic_iter : bool,False
            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
            receive in equal number of chances to flip.
        """

        cpucount=cpucount or self.nCpus
        if initial_sample is None:
            self.samples = self.rng.choice([-1.,1.],size=(sampleSize,self.n))
        else:
            self.samples = initial_sample
        self.E = np.array([ self.calc_e( s[None,:], self.theta ) for s in self.samples ])
       
        # Parallel sample.
        if not systematic_iter:
            def f(args):
                s, E, seed = args
                rng=np.random.RandomState(seed)
                for j in range(n_iters):
                    de = self._sample_metropolis( s, rng=rng )
                    E += de
                return s, E
        else:
            def f(args):
                s,E,seed = args
                rng=np.random.RandomState(seed)
                for j in range(n_iters):
                    de = self._sample_metropolis( s, rng=rng,flip_site=j%self.n )
                    E += de
                return s, E
        
        pool=mp.Pool(cpucount)
        self.samples,self.E = list(zip(*pool.map(f,list(zip(self.samples,
                                                self.E,
                                                np.random.randint(2**31-1,size=sampleSize))))))
        pool.close()

        self.samples = np.vstack(self.samples)
        self.E = np.concatenate(self.E)
    
    @staticmethod
    @njit
    def _jit_sample_metropolis(sample0, h, J, flip_site):
        """
        Metropolis sampling.
        """
        
        sample0[flip_site] *= -1
        de = -2*sample0[flip_site]*h[flip_site] - np.dot(2*J[flip_site], sample0[flip_site]*sample0)

        # Only accept flip if dE<=0 or probability exp(-dE)
        # Thus reject flip if dE>0 and with probability (1-exp(-dE))
        if de < 0:
            return de
        elif np.random.rand() > np.exp(-de):
            sample0[flip_site] *= -1.
            return 0.
        else:
            return de

    def _sample_metropolis(self, sample0, rng=None, flip_site=None):
        """
        Metropolis sampling.
        """

        rng = rng or self.rng
        flipSite = flip_site or rng.randint(sample0.size)
        sample0[flipSite] *= -1
        de = -2*sample0[flipSite]*self.h[flipSite] - 2*self.J[flipSite].dot(sample0[flipSite]*sample0)

        # Only accept flip if dE<=0 or probability exp(-dE)
        # Thus reject flip if dE>0 and with probability (1-exp(-dE))
        if de<0:
            return de
        elif rng.rand()>np.exp(-de):
            sample0[flipSite] *= -1.
            return 0.
        else:
            return de
#end FastMCIsing


class Metropolis(Sampler):
    def __init__(self, n, theta, calc_e,
                 n_cpus=None,
                 rng=None):
        """
        MC sample on Ising model with +/-1 formulation.

        Parameters
        ----------
        n : int
            Number of elements in system.
        theta : ndarray
            Vector of parameters in Hamiltonian.
        calc_e : function
            f( states, params )
        n_cpus : int,0
            If None, then will use all available CPUs.
        rng : np.random.RandomState,None
        """

        self.n = n
        self.theta = theta
        self.nCpus = n_cpus or mp.cpu_count()-1
        
        self.calc_e = calc_e

        if rng is None:
            self.rng=np.random.RandomState()
    
    def generate_samples(self,
                         sample_size,
                         n_iters=1000,
                         systematic_iter=False,
                         saveHistory=False,
                         initial_sample=None):
        """
        Generate Metropolis samples using a for loop.

        Parameters
        ----------
        sample_size : int
        n_iters : int,1000
        systematic_iter : bool,False
        saveHistory : bool,False
        initial_sample : ndarray,None

        Returns
        -------
        history : ndarray
        """

        if initial_sample is None:
            self.samples = self.rng.choice([-1.,1.],size=(sample_size,self.n))
        else: self.samples = initial_sample
        self.E = np.array([ self.calc_e( s[None,:], self.theta ) for s in self.samples ])

        if saveHistory:
            history=np.zeros((sample_size,n_iters+1))
            history[:,0]=self.E.ravel()

            if systematic_iter:
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self.samples[i], self.E[i], flip_site=j%self.n )
                        self.E[i] += de
                        history[i,j+1]=self.E[i]
            else:
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self.samples[i], self.E[i] )
                        self.E[i] += de
                        history[i,j+1]=self.E[i]
            return history
        else:
            if systematic_iter:
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self.samples[i], self.E[i], flip_site=j%self.n )
                        self.E[i] += de
            else:
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self.samples[i], self.E[i] )
                        self.E[i] += de

    def generate_samples_parallel(self,
                                  sample_size,
                                  n_iters=1000,
                                  cpucount=None,
                                  initial_sample=None,
                                  systematic_iter=False,
                                  ):
        """
        Generate samples in parallel and save them into self.samples and their energies into self.E.

        Parameters
        ----------
        sample_size : int
        n_iters : int,1000
        cpucount : int,None
        initial_sample : ndarray,None
        systematic_iter : bool,False
            Iterate through spins systematically instead of choosing them randomly.

        Returns
        -------
        None
        """

        cpucount = cpucount or self.nCpus
        if initial_sample is None:
            self.samples = self.rng.choice([-1.,1.], size=(sample_size,self.n))
        else:
            self.samples = initial_sample
        self.E = self.calc_e( self.samples, self.theta )
       
        # Parallel sample.
        if not systematic_iter:
            def f(args):
                s, E, seed = args
                rng = np.random.RandomState(seed)
                for j in range(n_iters):
                    de = self.sample_metropolis( s, E, rng=rng )
                    E += de
                return s, E
        else:
            def f(args):
                s, E, seed = args
                rng = np.random.RandomState(seed)
                for j in range(n_iters):
                    de = self.sample_metropolis( s, E, rng=rng, flip_site=j%self.n )
                    E += de
                return s, E
        
        pool = mp.Pool(cpucount)
        self.samples, self.E = list(zip(*pool.map(f,zip(self.samples,
                                                        self.E,
                                                        np.random.randint(2**31-1,size=sample_size)))))
        pool.close()

        self.samples = np.vstack(self.samples)
        self.E = np.concatenate(self.E)

    def generate_cond_samples(self,
                              sample_size,
                              fixed_subset,
                              burn_in=1000,
                              cpucount=None,
                              initial_sample=None,
                              systematic_iter=False,
                              parallel=True):
        """
        Generate samples from conditional distribution (while a subset of the spins are held fixed).
        Samples are generated in parallel.
        
        NOTE: There is a bug with multiprocess where many calls to the parallel sampling routine in
        a row leads to increasingly slow evaluation of the code.

        Parameters
        ----------
        sample_size : int
        fixed_subset : list of duples
            Each duple is the index of the spin and the value to fix it at.  These should be ordered
            by spin index.
        burn_in : int,1000
        cpucount : int,None
        initial_sample : ndarray,None
        systematic_iter : bool,False
            Iterate through spins systematically instead of choosing them randomly.
        """
        cpucount = cpucount or self.nCpus
        nSubset = self.n-len(fixed_subset)

        # Initialize sampler.
        if initial_sample is None:
            self.samples = self.rng.choice([-1.,1.],size=(sample_size,nSubset))
        else:
            self.samples = initial_sample
        
        # Redefine calc_e to calculate energy and putting back in the fixed spins.
        def cond_calc_e(state, theta):
            """
            Parameters
            ----------
            state : ndarray
                Free spins (not fixed).
            theta : ndarray
                Parameters.
            """

            fullstate = np.zeros((1,self.n))
            i0 = 0
            stateix = 0
            # Fill all spins in between fixed ones.
            for i,s in fixed_subset: 
                for ii in range(i0,i):
                    fullstate[0,ii] = state[0,stateix] 
                    stateix += 1
                fullstate[0,i] = s
                i0 = i+1
            # Any reamining spots to fill.
            for ii in range(i0,self.n):
                fullstate[0,ii] = state[0,stateix]
                stateix += 1
            return self.calc_e(fullstate,theta)
        self.E = np.array([ cond_calc_e( s[None,:], self.theta ) for s in self.samples ])
        
        # Parallel sample.
        if parallel:
            if not systematic_iter:
                def f(args):
                    s,E,seed = args
                    rng = np.random.RandomState(seed)
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=rng,calc_e=cond_calc_e )
                        E += de
                    return s,E
            else:
                def f(args):
                    s,E,seed=args
                    rng = np.random.RandomState(seed)
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=rng,flip_site=j%nSubset,calc_e=cond_calc_e )
                        E += de
                    return s,E
            
            #start = datetime.now()
            pool=mp.Pool(cpucount)
            #poolt = datetime.now()
            self.samples,self.E=list(zip(*pool.map(f,list(zip(self.samples,
                                                    self.E,
                                                    np.random.randint(0,2**31-1,size=sample_size))))))
            self.samples = np.vstack(self.samples)
            #samplet = datetime.now()
            pool.close()
            #poolcloset = datetime.now()

            #print "%1.1fs, %1.1fs, %1.1fs"%((poolt-start).total_seconds(),
            #                                (samplet-poolt).total_seconds(),
            #                                (poolcloset-samplet).total_seconds())
        else:
            rng = np.random.RandomState()
            
            if not systematic_iter:
                def f(args):
                    s,E = args
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=rng,calc_e=cond_calc_e )
                        E += de
                    return s,E
            else:
                def f(args):
                    s,E=args
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=rng,flip_site=j%nSubset,calc_e=cond_calc_e )
                        E += de
                    return s,E
           
            for i in range(len(self.samples)):
                s,E = f((self.samples[i],self.E[i]))
                self.samples[i] = s
                self.E[i] = E

        # Insert fixed spins back in.
        counter = 0
        for i,s in fixed_subset:
            if i==0:
                self.samples = np.insert(self.samples,list(range(i,self.samples.size,nSubset+counter)),s)
            else:
                self.samples = np.insert(self.samples,list(range(i,self.samples.size+1,nSubset+counter)),s)
            counter += 1
        self.samples = np.reshape(self.samples,(sample_size,self.n))
        self.E = np.concatenate(self.E)
        return self.samples,self.E

    def sample_metropolis(self, sample0, E0,
                          rng=None,
                          flip_site=None,
                          calc_e=None):
        """Metropolis sampling given an arbitrary sampling function.
        """

        rng = rng or self.rng
        flip_site = flip_site or rng.randint(sample0.size)
        calc_e = calc_e or self.calc_e

        sample0[flip_site] *= -1
        E1 = calc_e( sample0[None,:], self.theta )
        de = E1-E0

        # Only accept flip if dE<=0 or probability exp(-dE)
        # Thus reject flip if dE>0 and with probability (1-exp(-dE))
        if ( de>0 and (rng.rand()>np.exp(-de)) ):
            sample0[flip_site] *= -1.
            return np.zeros(1)
        else:
            return de
#end Metropolis


class HamiltonianMC(Sampler):
    def __init__(self, n, theta, calc_e, random_sample,
                 grad_e=None,
                 dt=.01,
                 leapfrogN=20,
                 nCpus=0):
        """
        NOTE: This has not been properly tested.

        Parameters
        ----------
        n : int
            Number of elements in system.
        theta : ndarray
            Vector of parameters in Hamiltonian.
        calc_e : function
            Must take in straight vector of numbers.
        grad_e : function,None
        dt : float,.01
            Momentum step length.
        leapfrogN : int,20
        """

        raise NotImplementedError("This hasn't been properly tested.")
        self.dt = dt
        self.leapfrogN = leapfrogN
        self.n = n
        self.theta = theta
        if nCpus is None:
            self.nCpus = mp.cpu_count()
        else:
            self.nCpus = nCpus
        
        self.calc_e = calc_e
        if grad_e is None:
            self.grad_e = Gradient(lambda x: calc_e(self.theta,x))
        else:
            self.grad_e = grad_e
        self.random_sample = random_sample
    
    def sample(self, x0, nBurn, saveHistory=False):
        """
        Get a single sample by MC sampling from this Hamiltonian. Slow method
        """

        if saveHistory:
            history = [x0]
        x = x0
        E = self.calc_e(self.theta,x0)
        g = self.grad_e(self.theta,x0)
        converged = False
        counter = 0
        
        while not converged:
            # Randomly sample momentum.
            p = np.random.normal(size=self.n)
            
            # Current Hamiltonian on joint space.
            H = p.dot(p)/2 + E
            
            # Leapfrog. Involves sandwiching gradient descent on q around x gradient descent.
            xnew = x
            gnew = g
            for t in range(self.leapfrogN):
                p -= self.dt*gnew/2
                xnew += self.dt*p
                
                gnew = self.grad_e(self.theta,xnew)
                p -= self.dt*gnew/2
            
            # Compute new energies.
            Enew = self.calc_e(self.theta,xnew)
            Hnew = p.dot(p)/2 + Enew
            dH = Hnew - H
            
            # MC sample.
            if dH<0 or np.random.rand()<np.exp(-dH):
                g = gnew
                E = Enew
                x = xnew
                
            if saveHistory:
                history.append(x.copy())
            if counter>nBurn:
                converged = True
            counter += 1
        
        if saveHistory:
            return x,history
        return x
    
    def generate_samples(self, nSamples, nBurn=100, fast=True, x0=None):
        """
        Generate nSamples from this Hamiltonian starting from random initial conditions from each sample.
        """

        if x0 is None:
            x0 = self.random_sample(nSamples)
        
        if self.nCpus==0:
            if fast:
                dt,leapfrogN,theta = self.dt,self.leapfrogN,self.theta
                @jit 
                def f(x0):
                    for i in range(nSamples):
                        r = np.random.normal(size=(nBurn,x0.shape[1]))
                        #r[:,::2] /= 2.
                        jit_sample( theta,x0[i],nBurn,dt,leapfrogN,
                                    r,
                                    np.random.rand(nBurn) )
                    return x0
                return f(x0)
            else:
                for i in range(nSamples):
                    x = self.sample(x0[i],nBurn)
                    samples[i,:] = x[:]
                return samples
        else:
            if fast:
                dt,leapfrogN,theta = self.dt,self.leapfrogN,self.theta
                @jit
                def f(x0):
                    rng = np.random.RandomState()
                    for i in range(nSamples):
                        x = jit_sample(theta,x0,nBurn,dt,leapfrogN,
                                       rng.normal(size=(nBurn,len(x0))),rng.rand(nBurn))
                        for j in range(len(x)):
                            samples[i,j] = x[j]
                    return samples
            else:
                def f(x0):
                    return self.sample(x0,nBurn)
            p = mp.Pool(self.nCpus)
            samples = np.vstack(( p.map(f,x0) ))
            p.close()
            return samples
#end HamiltonianMC


@njit
def jit_sample(theta, x0, nBurn, dt, leapfrogN, randNormal, randUnif):
    """
    Get a single sample by MC sampling from this Hamiltonian.

    Parameters
    ----------
    theta : ndarray
        Parameters
    x0 : ndarray
        Sample
    nBurn : int
    dt : float
    leapfrogN : int
    randNormal : ndarray
        nBurn x ndim
    randUnif : ndarray
        nBurn
    """

    x = x0
    E = calc_e(theta,x0)
    g = grad_e(theta,x0)
    # Randomly sample momenta.
    p = np.zeros_like(x0)

    for counter in range(nBurn):
        # Read in previously generated random momenta.
        for i in range(len(x0)):
            p[i] = randNormal[counter,i]

        # Current Hamiltonian on joint space.
        H = (p*p).sum()/2. + E
        
        # Leapfrog. Involves sandwiching gradient descent on q around x gradient descent.
        xnew = x
        gnew = g
        for t in range(leapfrogN):
            p -= dt*gnew/2.
            xnew += dt*p
            
            gnew = grad_e(theta,xnew)
            p -= dt*gnew/2.
        
        # Compute new energies.
        Enew = calc_e(theta,xnew)
        Hnew = (p*p).sum()/2. + Enew
        dH = Hnew - H
        
        # MC sample.
        if (dH<0) or (randUnif[counter]<np.exp(-dH)):
            g = gnew
            E = Enew
            x = xnew


class Heisenberg3DSampler(Sampler):
    """
    Simple MC Sampling from Heisenberg model with a lot of helpful functions.

    Methods
    -------
    generate_samples()
    equilibrate_samples()
    sample_metropolis()
    sample_energy_min()
    """
    def __init__(self, J, calc_e, random_sample):
        """
        NOTE: This has not been properly tested.

        Parameters
        ----------
        J : ndarray
            vector of coupling parameters
        calc_e : lambda
            Function for calculating energies of array of given states with args (J,states). States
            must be array with dimensions (nSamples,nSpins,3).  random_sample (lambda)
            Function for returning random samples with args (rng,n_samples).
        """

        raise NotImplementedError("This hasn't been properly tested.")
        self.J = J
        self.Jmat = squareform(J)
        self.Jrows = np.zeros((self.Jmat.shape[0],self.Jmat.shape[0]-1))
        for i in range(self.Jmat.shape[0]):
            ix = np.zeros((self.Jmat.shape[0]))==0
            ix[i] = False
            self.Jrows[i] = self.Jmat[i][ix]
        self.calc_e = calc_e
        self.random_sample = random_sample
        self.rng = np.random.RandomState()
        
    def generate_samples(self, nSamples, n_iters=100, **kwargs):
        # Initialize random samples
        samples = self.random_sample(self.rng,nSamples)
        self.equilibrate_samples( samples, n_iters,**kwargs )
        return samples
    
    def equilibrate_samples(self, samples, n_iters, method='mc', nCpus=0):
        if nCpus is None:
            nCpus = mp.cpu_count()
            print("Using all cores.")
        else:
            print("Using %d cores."%nCpus)
        if method=='mc':
            def sample_method(s,E):
                for i in range(n_iters):
                    _,E = self.sample_metropolis(s,E)
                return E
        else:
            raise Exception("Unsupported sampling method.")
        
        if nCpus==0:
            E = np.zeros((len(samples)))
            for i in range(len(samples)):
                E[i] = self.calc_e( self.J,[samples[i]] )
                E[i] = sample_method(samples[i],E[i])
            return
        else:
            p = mp.Pool(nCpus)
            def f(args):
                sample,J = args
                E = self.calc_e( J,[sample] )
                self.rng = np.random.RandomState()
                E = sample_method(sample,E)
                return E,sample
            E,samples[:] = list(zip( *p.map(f,list(zip(samples,[self.J]*len(samples)))) ))
            p.close()

    def sample_metropolis(self, oldState, E0):
        newState = self.sample_nearby_sample(oldState,sigma=.3)
        E1 = self.calc_e( self.J, [newState] )
        
        de = E1-E0
        if (de>0 and (self.rng.rand()>np.exp(-de))):
            return oldState,E0
        oldState[:] = newState[:]
        return newState,E1

    def sample_nearby_vector(self, v, nSamples=1, otheta=None, ophi=None, sigma=.1):
        """
        Sample random vector that is nearby. It is important how you choose the width sigma.
        NOTE: code might be simplified by using arctan2 instead of arctan
        
        Parameters
        ----------
        v : ndarray
            xyz vector about which to sample random vectors
        nSamples : int,1
            number of random samples
        otheta : float,None
            polar angle for v
        ophi : float,None
            azimuthal angle for v
        sigma : float,.1
            width of Gaussian about v
        """

        if otheta is None or ophi is None:
            r = np.sqrt( v[0]*v[0] + v[1]*v[1] )
            otheta = np.pi/2 - np.arctan( v[2]/r )
            if v[1]>=0:
                ophi = np.arccos( v[0]/r )
            else:
                ophi = 2*np.pi - np.arccos( v[0]/r )
        
        return jit_sample_nearby_vector( self.rng.randint(2**32-1),v,nSamples,otheta,ophi,sigma )

    def _sample_nearby_vector(self, v, nSamples=1, otheta=None, ophi=None, sigma=.1):
        """
        Deprecated: old slower way. Sample random vector that is nearby.
        
        Parameters
        ----------
        v : ndarray
            xyz vector about which to sample random vectors
        nSamples : int=1
            number of random samples
        otheta : float=None
            polar angle for v
        ophi : float=None
            azimuthal angle for v
        sigma : float=.1
            width of Gaussian about v
        """

        if otheta is None or ophi is None:
            r = np.sqrt( v[0]*v[0] + v[1]*v[1] )
            otheta = np.pi/2 - np.arctan( v[2]/r )
            if v[1]>=0:
                ophi = np.arccos( v[0]/r )
            else:
                ophi = 2*np.pi - np.arccos( v[0]/r )

        # Generate random vector with roughly Gaussian distribution on spherical surface about z-axis.
        if sigma==0:
            theta = 0
        else:
            theta = self.rng.normal(scale=sigma,size=nSamples)
        phi = self.rng.uniform(0,2*np.pi,size=nSamples)
        randv = np.vstack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T
        
        # Rotate into v frame using Rodrigues' formula.
        k = np.array([-np.sin(ophi),np.cos(ophi),0])
        return np.array([(r * np.cos(otheta) + np.cross(k,r)*np.sin(otheta) + k*np.dot(k,r)*(1-np.cos(otheta)))
                         for r in randv])

    def sample_nearby_sample(self, X, **kwargs):
        """
        Randomly move given state around for new metropolis sample.
        Question is whether it is more efficient to push only one of the many vectors around or all of them simultaneously.
        """

        return np.vstack([self.sample_nearby_vector(x,**kwargs) for x in X])

    def grad_E(self, X):
        """
        Gradient wrt theta and phi.

        Parameters
        ----------
        X : ndarray
            with dims (nSpins,2) with angles theta and phi
        """

        assert X.ndim==2
        
        g = np.zeros((X.shape[0],2))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i!=j:
                    g[i,0] += -( self.Jmat[i,j]*( np.cos(X[i,0])*np.sin(X[j,0])*np.cos(X[i,1]-X[j,1]) -
                                 np.sin(X[i,0])*np.cos(X[j,0]) ) )
                    g[i,1] += self.Jmat[i,j] * np.sin(X[j,0]) * np.sin(X[i,1]-X[j,1])
        return g

    def sample_energy_min(self,
                          nFixed=0,
                          rng=np.random.RandomState(),
                          initialState=None,
                          method='powell',
                          **kwargs):
        """
        Find local energy minimum given state in angular form. Angular representation makes it easy to be
        explicit about constraints on the vectors.
        
        Parameters
        ----------
        initialState : ndarray,None
            n_samples x n_features x 2
        nFixed : int,0
            Number of vectors that are fixed.
        """

        n = self.Jmat.shape[0]

        if initialState is None:
            initialState = np.zeros((1,n-nFixed,2))
            initialState[0,:,0] = rng.uniform(0,np.pi,size=n-nFixed)
            initialState[0,:,1] = rng.uniform(0,2*np.pi,size=n-nFixed)
        
        def f(sample):
            if np.any(sample<0):
                return np.inf
            sample = sample.reshape(1,n-nFixed,2)
            if np.any(sample[0,:,0]>np.pi):
                return np.inf

            sample = self.convert_to_xyz(sample)
            return self.calc_e(self.J,sample)

        if method=='fmin':
            return fmin(f,initialState.ravel())
        return minimize(f,initialState.ravel(),method=method,**kwargs)

    @classmethod
    def to_dict(cls,data,names):
        """
        Convenience function taking 3d array of of samples and arranging them into n x 3 arrays in a dictionary.
        """

        from collections import OrderedDict
        X = OrderedDict()
        for i,k in enumerate(names):
            X[k] = np.vstack([d[i,:] for d in data])
        return X
#end Heisenberg3dSampler

# ---------------------#
# Helper functions. #
# ---------------------#
@jit
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros((3))
    return cross_(vec1, vec2, result)

@njit
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    a1 = vec1[0]
    a2 = vec1[1]
    a3 = vec1[2]
    b1 = vec2[0]
    b2 = vec2[1]
    b3 = vec2[2]
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result

@njit
def jit_sample_nearby_vector(rseed,v,nSamples,otheta,ophi,sigma):
    np.random.seed(rseed)

    # Generate random vector with roughly Gaussian distribution on spherical surface about z-axis.
    if sigma==0:
        theta = np.zeros((1))
    else:
        theta = np.zeros((nSamples))
        for i in range(nSamples):
            theta[i] = np.random.randn()*sigma
    phi = np.zeros((nSamples))
    for i in range(nSamples):
        phi[i] = np.random.rand()*2*np.pi
    randv = np.zeros((nSamples,3))
    for i in range(nSamples):
        randv[i,0] = np.sin(theta[i])*np.cos(phi[i])
        randv[i,1] = np.sin(theta[i])*np.sin(phi[i])
        randv[i,2] = np.cos(theta[i])
    
    # Rotate into v frame using Rodrigues' formula.
    k = np.zeros((3))
    k[0] = -np.sin(ophi)
    k[1] = np.cos(ophi)
    rotatedv = np.zeros_like(randv)
    for i in range(len(randv)):
        rotatedv_ = (randv[i] * np.cos(otheta) + cross(k,randv[i])*np.sin(otheta) + k*np.dot(k,randv[i])*(1-np.cos(otheta)))
        for j in range(randv.shape[1]):
            rotatedv[i,j] = rotatedv_[j]
    return rotatedv

def check_e_logp(sample, calc_e):
    """
    Boltzmann type model with discrete state space should have E propto -logP. Calculate these quantities for
    comparison.

    Parameters
    ----------
    sample
    calc_e
    """

    from misc.utils import unique_rows

    uniqueix = unique_rows(sample,return_inverse=True)
    uniqStates = sample[unique_rows(sample)]

    stateCount = np.bincount(uniqueix)
    stateCount = stateCount/stateCount.sum()

    uniqueE = calc_e(uniqStates)
    return uniqStates,uniqueE,np.log(stateCount)

