"""Experimental / work-in-progress samplers.

This module collects three samplers whose implementations have not
been validated end-to-end. Constructing any of them currently raises
:class:`NotImplementedError`:

- :class:`SWIsing` — Swendsen-Wang cluster sampler for the Ising
  model (+/-1 basis).
- :class:`HamiltonianMC` — Hamiltonian Monte Carlo sampler whose
  default leapfrog helper is hardcoded for Heisenberg energies.
- :class:`Heisenberg3DSampler` — Heisenberg-3D MCMC sampler with
  spherical-vector moves.

Module-level helpers (``calc_e``, ``grad_e``, ``grad_e_theta``,
``cross``, ``cross_``, ``jit_sample``, ``jit_sample_nearby_vector``,
``pairwise_prod``, ``sample_bonds``, ``iter_cluster``,
``spec_cluster``, ``check_e_logp``) are kept here because they are
only consumed by the experimental classes above.
"""
import numpy as np
import multiprocess as mp
from numba import jit, njit
from numpy import sin, cos
from scipy.spatial.distance import squareform
from scipy.optimize import minimize, fmin

from ..samplers import Sampler
from ..utils import sub_to_ind, unique_rows


# --------------------------------------------------------------------------- #
# Heisenberg-model energy and gradient (used by HamiltonianMC.jit_sample and
# referenced by Heisenberg3DSampler).
# --------------------------------------------------------------------------- #
@njit
def calc_e(theta, x):
    """Heisenberg model energy.

    Parameters
    ----------
    theta : ndarray
        List of couplings Jij.
    x : ndarray
        List of angles (theta_0, phi_0, theta_1, phi_1, ...).
    """
    n = len(x)//2  # number of spins
    E = 0.
    k = 0
    for i in range(0, 2*(n-1), 2):
        for j in range(i+2, 2*n, 2):
            E += theta[k] * (sin(x[i])*sin(x[j]) *
                             (cos(x[i+1])*cos(x[j+1]) + sin(x[i+1])*sin(x[j+1])) +
                             cos(x[i])*cos(x[j]))
            k += 1
    return -E


@njit
def grad_e(theta, x):
    """Derivatives wrt the angles of the spins."""
    n = len(x)//2
    g = np.zeros((len(x)))

    for i in range(0, 2*n, 2):
        for j in range(0, 2*n, 2):
            if i != j:
                k = sub_to_ind(n, i//2, j//2)
                # Derivative wrt angle theta_i
                g[i] += theta[k] * (cos(x[i])*sin(x[j]) *
                                    (cos(x[i+1])*cos(x[j+1]) + sin(x[i+1])*sin(x[j+1])) -
                                    sin(x[i])*cos(x[j]))
                # Derivative wrt angle phi_i
                g[i+1] += theta[k] * (sin(x[i])*sin(x[j]) *
                                      (-sin(x[i+1])*cos(x[j+1]) + cos(x[i+1])*sin(x[j+1])))
    return -g


@njit
def grad_e_theta(theta, x):
    """Derivatives wrt the couplings theta."""
    n = len(x)//2
    g = np.zeros((len(theta)))
    k = 0
    for i in range(0, 2*(n-1), 2):
        for j in range(i+2, 2*n, 2):
            g[k] = (sin(x[i])*sin(x[j]) *
                    (cos(x[i+1])*cos(x[j+1]) + sin(x[i+1])*sin(x[j+1])) +
                    cos(x[i])*cos(x[j]))
            k += 1
    return -g


# --------------------------------------------------------------------------- #
# SWIsing
# --------------------------------------------------------------------------- #
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
    marked = set()
    clusters = []

    while initialSites:
        thisCluster = []
        newSites = [initialSites.pop()]
        marked.add(newSites[0])

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


def spec_cluster(L, exact=True):
    """
    Parameters
    ----------
    L : ndarray
        Graph Laplacian
    """
    from scipy.linalg import eig

    eig, v = eig(L)
    if exact:
        clusterix = np.nonzero(eig == 0)[0]
    else:
        clusterix = np.nonzero(np.isclose(eig, 0))[0]
    cluster = []
    for i in clusterix:
        cluster.append(np.nonzero(np.isclose(v[:, i], 0) == 0)[0])
    return cluster


# --------------------------------------------------------------------------- #
# HamiltonianMC
# --------------------------------------------------------------------------- #
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

    def generate_sample(self, nSamples, nBurn=100, fast=True, x0=None):
        """Generate nSamples from this Hamiltonian starting from random initial conditions
        from each sample.
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


# --------------------------------------------------------------------------- #
# Heisenberg3DSampler
# --------------------------------------------------------------------------- #
class Heisenberg3DSampler(Sampler):
    """
    Simple MC Sampling from Heisenberg model with a lot of helpful functions.

    Methods
    -------
    generate_sample()
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
            Function for calculating energies of array of given states with args
            (J,states). States must be array with dimensions (nSamples,nSpins,3).
            random_sample (lambda) Function for returning random samples with args
            (rng,n_samples).
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

    def generate_sample(self, nSamples, n_iters=100, **kwargs):
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
        """Randomly move given state around for new metropolis sample."""

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
        """Find local energy minimum given state in angular form."""

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
        """Arrange 3d samples into a dict of n x 3 arrays."""

        from collections import OrderedDict
        X = OrderedDict()
        for i,k in enumerate(names):
            X[k] = np.vstack([d[i,:] for d in data])
        return X


@jit
def cross(vec1, vec2):
    """Cross product of two 3D vectors."""
    result = np.zeros((3))
    return cross_(vec1, vec2, result)


@njit
def cross_(vec1, vec2, result):
    """Cross product of two 3D vectors (in-place)."""
    a1, a2, a3 = vec1[0], vec1[1], vec1[2]
    b1, b2, b3 = vec2[0], vec2[1], vec2[2]
    result[0] = a2*b3 - a3*b2
    result[1] = a3*b1 - a1*b3
    result[2] = a1*b2 - a2*b1
    return result


@njit
def jit_sample_nearby_vector(rseed, v, nSamples, otheta, ophi, sigma):
    np.random.seed(rseed)

    # Generate random vector with roughly Gaussian distribution on spherical surface about z-axis.
    if sigma == 0:
        theta = np.zeros((1))
    else:
        theta = np.zeros((nSamples))
        for i in range(nSamples):
            theta[i] = np.random.randn()*sigma
    phi = np.zeros((nSamples))
    for i in range(nSamples):
        phi[i] = np.random.rand()*2*np.pi
    randv = np.zeros((nSamples, 3))
    for i in range(nSamples):
        randv[i, 0] = np.sin(theta[i])*np.cos(phi[i])
        randv[i, 1] = np.sin(theta[i])*np.sin(phi[i])
        randv[i, 2] = np.cos(theta[i])

    # Rotate into v frame using Rodrigues' formula.
    k = np.zeros((3))
    k[0] = -np.sin(ophi)
    k[1] = np.cos(ophi)
    rotatedv = np.zeros_like(randv)
    for i in range(len(randv)):
        rotatedv_ = (randv[i] * np.cos(otheta) + cross(k, randv[i])*np.sin(otheta)
                     + k*np.dot(k, randv[i])*(1-np.cos(otheta)))
        for j in range(randv.shape[1]):
            rotatedv[i, j] = rotatedv_[j]
    return rotatedv


def check_e_logp(sample, calc_e):
    """Boltzmann-type model with discrete state space should have E proportional
    to -logP. Calculate these for comparison.
    """
    uniqueix = unique_rows(sample, return_inverse=True)
    uniqStates = sample[unique_rows(sample)]

    stateCount = np.bincount(uniqueix)
    stateCount = stateCount/stateCount.sum()

    uniqueE = calc_e(uniqStates)
    return uniqStates, uniqueE, np.log(stateCount)
