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
from numba import jit,njit,float64,int64
from numpy import sin,cos,exp
from scipy.spatial.distance import squareform
import multiprocess as mp
from .utils import *
from datetime import datetime
from multiprocess import Pool, cpu_count
from warnings import warn


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
@njit
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

@njit
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

@njit
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

@njit
def pairwise_prod(state):
    counter = 0
    n = len(state)
    prod = np.zeros((n*(n-1)//2))
    for i in range(n-1):
        for j in range(i+1,n):
            prod[counter] = state[i]*state[j]
            counter += 1
    return prod
        
@njit
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


class ParallelTempering(Sampler):
    def __init__(self, n, theta, calc_e, n_replicas,
                 Tbds=(1.,3.),
                 sample_size=1000,
                 replica_burnin=None,
                 rep_ex_burnin=None,
                 n_cpus=None,
                 rng=None):
        """
        Run multiple replicas in parallel at different temperatures using Metropolis sampling to
        equilibrate.

        Hukushima, K, and K Nemoto. “Exchange Monte Carlo Method and Application to Spin Glass
        Simulations.” Journal of the Physical Society of Japan 65 (1996): 1604–1608.

        Parameters
        ----------
        n : int
            Number of elements in system.
        theta : ndarray
            Concatenated vector of the field hi and couplings Jij. They should be ordered in
            ascending order 0<=i<n and for Jij in order of 0<=i<j<n.
        calc_e : function
            For calculating energy.
        n_replicas : int
            Number of replicas.
        Tbds : duple, (1,3)
            Lowest and highest temperatures to start with. Lowest temperature will not change.
        sample_size : int, 1000
        replica_burnin : int, n*50
            Default number of burn in iterations for each replica when first initialising (from completely
            uniform distribution).
        rep_ex_burnin : int, n*10
            Default number of Metropolis steps after exchange.
        n_cpus : int, 0
            If None, then will use all available CPUs minus 1.
        rng : RandomState, None
        """
        
        assert Tbds[0]<Tbds[1] and Tbds[0]>0
        assert n_replicas>1

        self.n = n
        self.theta = theta
        self.calc_e = calc_e
        self.nReplicas = n_replicas
        self.nCpus = n_cpus or cpu_count()-1
        self.sampleSize = sample_size
        self.samples = None
        self.replicaBurnin = replica_burnin or n*50
        self.repExBurnin = rep_ex_burnin or n*10
        self.rng = rng or np.random.RandomState()
       
        self.beta = self.initialize_beta(1/Tbds[1], 1/Tbds[0], n_replicas)
        self.setup_replicas()
        assert (np.diff(self.beta)>0).all(), self.beta
    
    def update_replica_parameters(self):
        """
        Update parameters for each replica. Remember that the parameters include the factor of beta.
        """

        for b,rep in zip(self.beta, self.replicas):
            rep.theta = self.theta*b

    def setup_replicas(self):
        """
        Initialise a set of replicas at different temperatures using the Metropolis algorithm and optimize the
        temperatures. Replicas are burned in and ready to sample.
        """

        self.replicas = []
        for i,b in enumerate(self.beta):
            self.replicas.append( Metropolis(self.n, self.theta*b, self.calc_e, n_cpus=1) )
            # give each replica an index
            self.replicas[i].index = i
        self.burn_in_replicas()

        self.optimize_beta(10, self.n*10)
        self.burn_in_replicas()

    def burn_in_replicas(self, pool=None, close_pool=True, n_iters=None):
        """Run each replica separately.
        
        Parameters
        ----------
        pool : multiprocess.Pool, None
        close_pool : bool, True
            If True, call pool.close() at end.
        n_iters : int, None
            Default value is self.replicaBurnin.
        """
        
        n_iters = n_iters or self.replicaBurnin

        pool = pool or Pool(self.nCpus)
        def f(args):
            rep, nIters = args
            rep.generate_samples(1, n_iters=nIters, systematic_iter=True)
            return rep
        self.replicas = pool.map(f, zip(self.replicas, [n_iters]*len(self.replicas)))

        if close_pool:
            pool.close()

    def burn_and_exchange(self, pool):
        """
        Parameters
        ----------
        pool : mp.multiprocess.Pool
        """
        
        self.burn_in_replicas(pool=pool, close_pool=False, n_iters=self.repExBurnin)
        for i in range(self.nReplicas-1):
            exchangeProb = self._acceptance_ratio(1, 0, pairs=[(i,i+1)])
            if self.rng.rand()<exchangeProb:
                # swap replicas (only need to swap samples)
                temp = self.replicas[i].samples
                self.replicas[i].samples = self.replicas[i+1].samples
                self.replicas[i+1].samples = temp

                temp = self.replicas[i]._samples
                self.replicas[i]._samples = self.replicas[i+1]._samples
                self.replicas[i+1]._samples = temp

                temp = self.replicas[i].index
                self.replicas[i].index = self.replicas[i+1].index
                self.replicas[i+1].index = temp

                # must recalculate energies with new temperature
                temp = self.replicas[i].E
                self.replicas[i].E = self.replicas[i+1].E/self.beta[i+1]*self.beta[i]
                self.replicas[i+1].E = temp/self.beta[i]*self.beta[i+1]

    def generate_samples(self,
                         sample_size,
                         save_exchange_trajectory=False):
        """
        Burn in, run replica exchange simulation, then sample.

        Parameters
        ----------
        sample_size : int
            Number of samples to take for each replica.
        save_exchange_trajectory : bool, False
            If True, keep track of the location of each replica in beta space and return the history.

        Returns
        -------
        ndarray, optional
            Trajectory of each replica through beta space. Each row is tells where each index is located in
            beta space.
        """
        
        self.samples = [np.zeros((sample_size,self.n), dtype=int) for i in range(self.nReplicas)]
        pool = Pool(self.nCpus)
        
        if save_exchange_trajectory:
            replicaIndexHistory = np.zeros((sample_size, self.nReplicas), dtype=int)

            for i in range(sample_size):
                self.burn_and_exchange(pool)

                # save samples
                for j in range(self.nReplicas):
                    self.samples[j][i,:] = self.replicas[j].samples[0,:]
                    replicaIndexHistory[i,j] = self.replicas[j].index
            pool.close()

            return replicaIndexHistory

        else:
            for i in range(sample_size):
                self.burn_and_exchange(pool)

                # save samples
                for j in range(self.nReplicas):
                    self.samples[j][i,:] = self.replicas[j].samples[0,:]
            pool.close()
    
    @staticmethod
    def initialize_beta(b0, b1, n_replicas):
        """Use linear interpolation of temperature range."""
        return np.linspace(b0, b1, n_replicas)
    
    @staticmethod
    def iterate_beta(beta, acceptance_ratio):
        """
        Apply algorithm from Hukushima but reversed to maintain one replica at T=1.

        Parameters
        ----------
        beta : ndarray
            Inverse temperature.
        acceptance_ratio : ndarray
            Estimate of acceptance ratio.

        Returns
        -------
        ndarray
            New beta.
        """
        
        assert (len(acceptance_ratio)+1)==len(beta)
        assert (acceptance_ratio<=1).all()
        
        newBeta = beta.copy()
        avg = acceptance_ratio.mean()
        for i in range(len(beta)-1, 0, -1):
            newBeta[i-1] = newBeta[i] - (beta[i]-beta[i-1]) * acceptance_ratio[i-1]/avg
        return newBeta

    def optimize_beta(self,
                      n_samples,
                      n_iters,
                      tol=.01,
                      max_iter=10):
        """
        Find suitable temperature range for replicas. Sets self.beta.

        Parameters
        ----------
        n_samples : int
            Number of samples to use to estimate acceptance ratio. Acceptance ratio is estimated as the
            average of these samples.
        n_iters : int
            Number of sampling iterations for each replica.
        tol : float, .1
            Average change in beta to reach before stopping.
        max_iter : int, 10
            Number of times to iterate algorithm for beta. Each iteration involves sampling from replicas.
        """
        
        beta = self.beta
        oldBeta = np.zeros_like(beta) + np.inf

        counter = 0
        while counter<max_iter and tol<np.abs(oldBeta-beta).mean():
            acceptanceRatio = self._acceptance_ratio(n_samples, n_iters)
            oldBeta = beta
            beta = self.iterate_beta(beta, acceptanceRatio)
            
            # change beta in the replicas for sampling with them
            self.beta = beta
            self.update_replica_parameters()

            counter += 1
            #print(beta, acceptanceRatio)
        
        if counter==max_iter:
            print("Optimization for beta did not converge.")

        # TODO: smooth by spline interpolation

        self.beta = beta
            
    def _acceptance_ratio(self, n_samples, n_iters, pairs=None):
        """Estimate acceptance ratio as an average over multiple Metropolis samples.
        
        Parameters
        ----------
        n_samples : int
            Number of Metropolis samples to use for averaging the ratio.
        n_iters : int
            Number of MC steps between samples. Bigger is better.
        pairs : list of duples, None
            Pairs for which to compute the acceptance ratio. If not given, all pairs are compared.

        Returns
        -------
        ndarray
            Estimate of acceptance ratio for each pair.
        """
        assert all([len(r.samples)==1 for r in self.replicas])
        
        if pairs is None:
            pairs = [(i,i+1) for i in range(self.nReplicas-1)]
        acceptanceRatio = np.zeros((len(pairs),n_samples))

        # estimate acceptance probabilities
        if n_iters>0:
            pool = Pool(self.nCpus)
            for i in range(n_samples):
                self.burn_in_replicas(pool=pool, close_pool=False, n_iters=n_iters)
                
                for j,p in enumerate(pairs):
                    # must divide out self.beta to get energies
                    dE = self.replicas[p[0]].E/self.beta[p[0]] - self.replicas[p[1]].E/self.beta[p[1]]
                    acceptanceRatio[j,i] = min(1, np.exp( dE * (self.beta[p[0]]-self.beta[p[1]]) ))
            pool.close()
        else:
            for j,p in enumerate(pairs):
                # must divide out self.beta to get energies
                dE = self.replicas[p[0]].E/self.beta[p[0]] - self.replicas[p[1]].E/self.beta[p[1]]
                acceptanceRatio[j,0] = min(1, np.exp( dE * (self.beta[p[0]]-self.beta[p[1]]) ))

        return acceptanceRatio.mean(1) 
#end ParallelTempering


#class FastMCIsing(Sampler):
#    def __init__(self, n, theta,
#                 n_cpus=None,
#                 rng=None):
#        """
#        MC sample on Ising model with +/-1 formulation. Fast metropolis sampling by assuming form of
#        Hamiltonian is an Ising model.
#
#        Parameters
#        ----------
#        n : int
#            Number of elements in system.
#        theta : ndarray
#            Concatenated vector of the field hi and couplings Jij. They should be ordered in
#            ascending order 0<=i<n and for Jij in order of 0<=i<j<n.
#        n_cpus : int, 0
#            If None, then will use all available CPUs minus 1.
#        rng : RandomState, None
#        """
#
#        warn("At the moment, Metropolis is much faster for sampling.")
#        self.n = n
#        self.update_parameters(theta)
#        self.nCpus = n_cpus or mp.cpu_count()-1
#
#        self.calc_e, calc_observables, mchApproximation = define_ising_helper_functions()
#        self.sample_metropolis = _jit_sample_metropolis
#        self.rng = rng or np.random.RandomState()
#        self._samples = None
#
#    def update_parameters(self, theta):
#        self.theta = theta
#        self.h, self.J = theta[:self.n], squareform(theta[self.n:])
#
#    def generate_samples(self,
#                         sample_size,
#                         n_iters=1000,
#                         saveHistory=False,
#                         initial_sample=None,
#                         systematic_iter=False):
#        """
#        Generate Metropolis samples using a for loop and save samples in self.samples.
#
#        Parameters
#        ----------
#        sample_size : int
#            Number of samples to take.
#        n_iters : int, 1000
#            Number of iterations to run Metropolis sampler.
#        saveHistory : bool, False
#            If True, the energy of the system after each Metropolis sampling step will be recorded.
#            This can be useful for making sure that the system has reached equilibrium.
#        initial_sample : ndarray, None
#            If this is given, then this is used as the starting point for the Markov chain instead
#            of a random sample.
#        systematic_iter : bool, False
#            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
#            receive in equal number of chances to flip.
#        """
#
#        sample_metropolis = self.sample_metropolis  # alias
#
#        if (initial_sample is None and
#            (self._samples is None or len(self._samples)!=1)):
#            self._samples = self.rng.choice([-1,1], size=(1, self.n))
#        elif not initial_sample is None:
#            msg = "Sequential sample generation requires initial sample of dim (1, n)."
#            assert np.array_equal((1,self.n), initial_sample.shape), msg
#            self._samples = initial_sample.astype(int)
#
#        E = self.calc_e( self._samples, self.theta )
#        self.E = np.zeros(sample_size)
#        self.samples = np.zeros((sample_size, self.n), dtype=int)
#        h, J = self.h, self.J
#        n = self.n
#
#        if systematic_iter:
#            @jit(forceobj=True)
#            def get_ix(j, rng):
#                return j%n
#        else:
#            @jit(forceobj=True)
#            def get_ix(j, rng):
#                return rng.randint(n)
#
#        if saveHistory:
#            @jit('float64[:,:]()', locals={'_samples':int64[:,:], 'E':float64[:]}, forceobj=True)
#            def sample(seed, E=E, _samples=self._samples):
#                # set rng for jit environment
#                np.random.seed(seed)
#
#                history = np.zeros(sample_size*n_iters+1)
#                history[0] = E
#                counter = 1
#                for i in range(sample_size):
#                    for j in range(n_iters):
#                        de = sample_metropolis( _samples[0], h, J, get_ix(j, np.random), np.random )
#                        E += de
#                        history[counter] = E
#                        counter += 1
#                    self.E[i] = E
#                    self.samples[i,:] = _samples[:]
#                return history
#        else:
#            @jit(locals={'_samples':int64[:,:], 'E':float64[:]}, forceobj=True)
#            def sample(seed, E=E, _samples=self._samples):
#                # set rng for jit environment
#                np.random.seed(seed)
#
#                for i in range(sample_size):
#                    for j in range(n_iters):
#                        de = sample_metropolis( _samples[0], h, J, get_ix(j, np.random), np.random )
#                        E += de
#                    self.E[i] = E
#                    self.samples[i,:] = _samples[:]
#        return sample(self.rng.randint(2**32-1))
#
#    def generate_samples_parallel(self,
#                                  sample_size,
#                                  n_iters=1000,
#                                  n_cpus=None,
#                                  initial_sample=None,
#                                  systematic_iter=False):
#        """
#        Metropolis sample multiple states in parallel and save them into self.samples.
#
#        Parameters
#        ----------
#        sample_size : int
#            Number of samples to take.
#        n_iters : int, 1000
#            Number of iterations to run Metropolis sampler.
#        initial_sample : ndarray, None
#            If this is given, then this is used as the starting point for the Markov chain instead
#            of a random sample.
#        systematic_iter : bool, False
#            If True, will iterate through each spin in a fixed sequence. This ensures that all spins
#            receive in equal number of chances to flip.
#        """
#
#        n_cpus = self.nCpus  # alias
#        assert n_cpus>=2
#        assert sample_size>n_cpus, "Parallelization only helps if many samples are generated per thread."
#        if (initial_sample is None and
#            (self._samples is None or len(self._samples)!=n_cpus)):
#            self._samples = self.rng.choice([-1,1], size=(n_cpus, self.n))
#        elif not initial_sample is None:
#            assert np.array_equal((n_cpus,self.n), initial_sample.shape), "initial_sample wrong  size"
#            self._samples = initial_sample.astype(int)
#
#        n_cpus = n_cpus or self.nCpus
#        sample_metropolis = _jit_sample_metropolis
#        h, J = self.h, self.J
#        n = self.n
#        calc_e = self.calc_e
#        theta = self.theta
#
#        if systematic_iter:
#            @jit(forceobj=True)
#            def get_ix(j, rng):
#                return j%n
#        else:
#            @jit(forceobj=True)
#            def get_ix(j, rng):
#                return rng.randint(n)
#
#        @njit(locals={'seed':int64, 'theta':float64[:], 's':int64[:,:], 'E':float64[:]})
#        def f(args):
#            seed, theta, s = args
#            np.random.seed(seed)
#
#            E = calc_e(s, theta)
#
#            for j in range(n_iters):
#                de = sample_metropolis( s[0], h, J, get_ix(j, np.random), np.random )
#                E += de
#            return s, E
#
#        args=((self.rng.randint(2**32-1),theta,self._samples[i][None,:]) for i in range(n_cpus))
#
#        # run sampling
#        pool = mp.Pool(n_cpus)
#        self.samples, self.E = list(zip(*pool.map(f, args)))
#        pool.close()
#
#        # save results of sampling into instance data members
#        self.samples = np.vstack(self.samples)
#        self.E = np.array(self.E)
#
#    def _sample_metropolis(self, sample0, rng=None, flip_site=None):
#        """
#        Metropolis sampling.
#        """
#
#        rng = rng or self.rng
#        flipSite = flip_site or rng.randint(sample0.size)
#        sample0[flipSite] *= -1
#        de = -2*sample0[flipSite]*self.h[flipSite] - 2*self.J[flipSite].dot(sample0[flipSite]*sample0)
#
#        # Only accept flip if dE<=0 or probability exp(-dE)
#        # Thus reject flip if dE>0 and with probability (1-exp(-dE))
#        if de<0:
#            return de
#        elif rng.rand()>np.exp(-de):
#            sample0[flipSite] *= -1
#            return 0.
#        else:
#            return de
#end FastMCIsing

@jit(forceobj=True)
def _jit_sample_metropolis(sample0, h, J, flip_site, rng):
    """
    Metropolis sampling.
    """
    
    sample0[flip_site] *= -1
    de = -2*sample0[flip_site]*h[flip_site] - np.dot(2*J[flip_site], 1.*sample0[flip_site]*sample0)

    # Only accept flip if dE<=0 or probability exp(-dE)
    # Thus reject flip if dE>0 and with probability (1-exp(-dE))
    if de < 0:
        return de
    elif rng.rand() > np.exp(-de):
        sample0[flip_site] *= -1
        return 0.
    else:
        return de


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
        n_cpus : int
            If None, then will use all available CPUs.
        rng : np.random.RandomState
            Random number generator.
        """

        self.n = n
        self.theta = theta
        self.nCpus = n_cpus or mp.cpu_count()-1
        self.calc_e = calc_e
        self.rng = rng or np.random.RandomState()
        self._samples = None
    
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
            Number of samples.
        n_iters : int, 1000
            Number of iterations to run the sampler floor.
        systematic_iter : bool, False
            If True, iterate through each element of system by increment index by one. 
        saveHistory : bool, False
            If True, also save the energy of each sample at each sampling step.
        initial_sample : ndarray, None
            Start with this sample (i.e. to avoid warming up). Otherwise, self._samples is the initial sample.

        Returns
        -------
        ndarray, optional
            Saved array of energies at each sampling step.
        """
        
        assert self.nCpus<=1, "Instantiate another instance for sequential sampling."
        if (initial_sample is None and
            (self._samples is None or len(self._samples)!=sample_size)):
            self._samples = self.rng.choice([-1,1], size=(1, self.n))
        elif not initial_sample is None:
            msg = "Initial sample can only be one state for sequential sampling."
            assert np.array_equal((1,self.n), initial_sample.shape), msg
            self._samples = initial_sample.astype(int)

        E = self.calc_e( self._samples, self.theta )
        self.samples = np.zeros((sample_size, self.n), dtype=int)
        self.E = np.zeros(sample_size)

        if saveHistory:
            history = np.zeros(sample_size*n_iters+1)

            if systematic_iter:
                counter = 0
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self._samples[0], E, flip_site=j%self.n, rng=self.rng )
                        E += de
                        history[counter] = E
                        counter += 1
                    self.samples[i,:] = self._samples[:]
                    self.E[i] = E
            else:
                counter = 0
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self._samples[0], E, rng=self.rng )
                        E += de
                        history[counter]=E
                        counter += 1
                    self.samples[i,:] = self._samples[:]
                    self.E[i] = E
            return history
        else:
            if systematic_iter:
                counter = 0
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self._samples[0], E, flip_site=j%self.n, rng=self.rng )
                        E += de
                        counter +=1
                    self.samples[i,:] = self._samples[:]
                    self.E[i] = E
            else:
                counter = 0
                for i in range(sample_size):
                    for j in range(n_iters):
                        de = self.sample_metropolis( self._samples[0], E, rng=self.rng )
                        E += de
                        counter += 1
                    self.samples[i,:] = self._samples[:]
                    self.E[i] = E

    def generate_samples_parallel(self,
                                  sample_size,
                                  n_iters=1000,
                                  initial_sample=None,
                                  systematic_iter=False):
        """
        Generate samples in parallel. Each replica in self._samples runs on its own thread and a 
        sample is generated every n_iters.

        Parameters
        ----------
        sample_size : int
            Number of samples.
        n_iters : int, 1000
            Number of iterations between taking a random sample.
        initial_sample : ndarray, None
            Starting set of replicas otherwise self._samples is used.
        systematic_iter : bool, False
            If True, iterate through spins systematically instead of choosing them randomly.
        """
        
        n_cpus = self.nCpus  # alias
        assert n_cpus>=2, "Instantiate another instance for parallel sampling."
        assert sample_size>n_cpus, "Parallelization only helps if many samples are generated per thread."
        if (initial_sample is None and
            (self._samples is None or len(self._samples)!=n_cpus)):
            self._samples = self.rng.choice([-1,1], size=(n_cpus, self.n))
        elif not initial_sample is None:
            assert np.array_equal((n_cpus,self.n), initial_sample.shape), "initial_sample wrong  size"
            self._samples = initial_sample.astype(int)

        E = self.calc_e( self._samples, self.theta )
        self.samples = None  # delete this to speed up pickling for multiprocess
       
        # Parallel sample. Each thread needs to return sample_size/n_cpus samples.
        if not systematic_iter:
            def f(args):
                s, E, nSamples, seed = args
                rng = np.random.RandomState(seed)
                samples = np.zeros((nSamples, self.n), dtype=int)
                for i in range(nSamples):
                    for j in range(n_iters):
                        de = self.sample_metropolis( s, E, rng=rng )
                        E += de
                    samples[i,:] = s[:]
                return samples, s, E
        else:
            def f(args):
                s, E, nSamples, seed = args
                rng = np.random.RandomState(seed)
                samples = np.zeros((nSamples, self.n), dtype=int)
                for i in range(nSamples):
                    for j in range(n_iters):
                        de = self.sample_metropolis( s, E, rng=rng, flip_site=j%self.n )
                        E += de
                    samples[i,:] = s[:]
                return samples, s, E
        
        pool = mp.Pool(n_cpus)
        self.samples, self._samples, self.E = list(zip(*pool.map(f,zip(self._samples,
                                                        E,
                                                        [int(np.ceil(sample_size/n_cpus))]*n_cpus,
                                                        self.rng.randint(2**31-1,size=n_cpus)))))
        pool.close()
        
        self.samples = np.vstack(self.samples)[:sample_size]
        self._samples = np.vstack(self._samples)

    def generate_cond_samples(self,
                              sample_size,
                              fixed_subset,
                              burn_in=1000,
                              n_cpus=None,
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
        burn_in : int
            Burn in.
        n_cpus : int
            Number of cpus to use.
        initial_sample : ndarray
            Option to set initial random sample.
        systematic_iter : bool
            Iterate through spins systematically instead of choosing them randomly.
        parallel : bool
            If True, use parallelized routine.

        Returns
        -------
        ndarray
            Samples from distribution.
        ndarray
            Energy of each sample.
        """

        n_cpus = n_cpus or self.nCpus
        nSubset = self.n-len(fixed_subset)

        # Initialize sampler.
        if initial_sample is None:
            self.samples = self.rng.choice([-1,1], size=(sample_size,nSubset))
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

            Returns
            -------
            float
                Energy of state with fixed spins.
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
            
            # avoid pickling a copy of self.samples into every thread
            samples = self.samples
            self.samples = None

            #start = datetime.now()
            pool=mp.Pool(n_cpus)
            #poolt = datetime.now()
            args = zip(samples, self.E, self.rng.randint(0, 2**31-1, size=sample_size))
            self.samples, self.E=list(zip(*pool.map(f, args)))
            self.samples = np.vstack(self.samples)
            #samplet = datetime.now()
            pool.close()
            #poolcloset = datetime.now()

            #print "%1.1fs, %1.1fs, %1.1fs"%((poolt-start).total_seconds(),
            #                                (samplet-poolt).total_seconds(),
            #                                (poolcloset-samplet).total_seconds())
        else:
            if not systematic_iter:
                def f(args):
                    s, E = args
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=self.rng,calc_e=cond_calc_e )
                        E += de
                    return s, E
            else:
                def f(args):
                    s, E=args
                    for j in range(burn_in):
                        de = self.sample_metropolis( s,E,rng=self.rng,flip_site=j%nSubset,calc_e=cond_calc_e )
                        E += de
                    return s, E
           
            for i in range(len(self.samples)):
                s, E = f((self.samples[i],self.E[i]))
                self.samples[i] = s
                self.E[i] = E

        # Insert fixed spins back in.
        counter = 0
        for i,s in fixed_subset:
            if i==0:
                self.samples = np.insert(self.samples, list(range(i,self.samples.size,nSubset+counter)), s)
            else:
                self.samples = np.insert(self.samples, list(range(i,self.samples.size+1,nSubset+counter)), s)
            counter += 1
        self.samples = np.reshape(self.samples, (sample_size,self.n))
        self.E = np.concatenate(self.E)
        return self.samples, self.E

    def sample_metropolis(self, sample0, E0,
                          rng=None,
                          flip_site=None,
                          calc_e=None):
        """Metropolis sampling given an arbitrary sampling function.

        Parameters
        ----------
        sample0 : ndarray
            Sample to start with. Passed by ref and changed.
        E0 : ndarray
            Initial energy of state.
        rng : np.random.RandomState
            Random number generator.
        flip_site : int
            Site to flip.
        calc_e : function
            If another function to calculate energy should be used
            
        Returns
        -------
        float
            delta energy.
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
            sample0[flip_site] *= -1
            return np.zeros(1, dtype=int)
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

