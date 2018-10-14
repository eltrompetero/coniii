"""
Swendson & Wang Replica Monte Carlo for general ising model.

Author: Colin Clement
Date: 2016-6-11

This implementation follows: arXiv:cond-mat/0407273v1
The only modification is a general implementation for any
graph with two-spin interactions.

"""

import numpy as np
import scipy.sparse as sps
from itertools import islice
from collections import defaultdict
from functools import wraps
from subprocess import check_output
from math import exp
from scipy.spatial.distance import squareform
from functools import reduce
try:
    import pickle as pickle
except ImportError:
    import pickle
try:
    from numba import jit
    hasnumba = True
except ImportError as perror:
    hasnumba = False


class GeneralSpinModel(object):
    """Swendsen and Wang replica Monte Carlo for ising model with arbitrary
    couplings and interaction graphs"""
    def __init__(self, J, templist = [1.], observers = None, sparse = True, **kwargs):
        """
            Monte carlo simulation for generalized ising couplings. 
            Parameters:
                J: coupling array of size (N_spins, N_spins), ideally sparse 
                templist: list of temperatures to simulate, should be ordered.
                observers : list of observer functions
            kwargs:
                seed : Seed for initializing random number generator
                therm : Initially thermalize for integer number of steps provided
            Kwargs:
        """
        if len(J.shape)==1:
            self.J = squareform(J)
        elif (J-J.T).sum() > 0:
            raise RuntimeError
        else:
            self.J = J
        if sparse:
            self.J = sps.csr_matrix(self.J)
        self.N = J.shape[0] 
        self.templist = templist
        self.Ntemps = len(templist)
        self.time = 0 #increments each MC step
        self.seed = kwargs.get('seed', 92089)
        self.therm = kwargs.get('therm', 0)
        self.observers = observers or []
       
        # Data structures of J matrix that are more efficient
        self.bonds = defaultdict(set)
        self.neighborsets = [neighbors(i, self.J) for i in range(self.N)]
        self.neighborlist = [] 
        self.neighborbonds = []
        self.bondmap = {}
        for i, neighs in enumerate(self.neighborsets):
            self.neighborlist += [np.array(list(neighs))]
            self.neighborbonds += [self.J[i, list(neighs)].toarray().flatten()]
            for n in neighs:
                if i < n: self.bonds[i].update(set([(i,n)]))
                else: self.bonds[i].update(set([(n,i)]))
                self.bondmap[(i,n)] = self.J[i,n]

        self.cumSpinPair = np.zeros((self.N, self.N))
        self.pairtime = 0.
        self._seeded_rand() #creates self._my_rng random number generator
        self.spinarray = 2*(self._my_rng.rand(len(self.templist), self.N)>0.5) - 1
        
        if self.therm: self.thermalize(self.therm)

    def __repr__(self):
        formatstring = "<{cls}(Ntemps={Ntemps}, N={N}, time = {time}, seed = {seed})>"
        return formatstring.format(cls=self.__class__.__name__, **self.__dict__)

    def __getstate__(self):
        """
        for pickling, observers are not pickleable
        """
        gooddict = {key:value for key,value in self.__dict__.items() if not
                    key=='observers'}
        gooddict['observers'] = []
        return gooddict

    def __iter__(self):
        """ Make an instance of this class an iterator. See the next method."""
        return self
   
    def _seeded_rand(self):
        """Set up random number generator"""
        self._my_rng = np.random.RandomState()
        self._my_rng.seed(self.seed)
    
    def _increment_time(self):
        """Count time steps (total MC sweeps)"""
        self.time += 1
        self.notify_observers() #always notify, observer decides if time is right

    def thermalize(self, relax_time):
        """
        performs relax_time (int) monte carlo steps
        """
        for state in islice(self,relax_time):
            pass

    def energy(self, s, J = None):
        """Return the energy of the current state"""
        J = J or self.J
        return -s.dot(J.dot(s))/2.

    def _oneSweep(self, s, T):
        """
        One Monte Carlo step for each spin in s
        input:
            s : spin array of length self.N
            T : temperature float
        """
        randtest = self._my_rng.rand(self.N)
        for i, test in enumerate(randtest):
            dE = 2*s[i]*(self.J[i].dot(s))
            if dE < 0 or test <= exp(-dE/T): s[i] *= -1

    def clusterStep(self, s0, T0, s1, T1):
        """
        Find clusters of constant overlap between
        spin configurations s0 and s1, perform one sweep
        of replica cluster Monte Carlo.
        input:
            s0: spin configuration 0
            T0: Temperature of spin configuration 0
            s1: spin configuration 1
            T1: Temperature of spin configuration 1
        """
        tau = s0*s1
        cls = getClusters(tau, self.neighborsets)
        k_ab = (1./T0 - 1./T1)*self.getClusterCouplings(cls, s0)
        eta = np.ones(len(cls))
        randtest = self._my_rng.rand(len(cls))
        for i, test in enumerate(randtest):
            dE = 2*eta[i]*(k_ab[i].dot(eta))
            if dE < 0 or test <= exp(-dE): 
                eta[i] *= -1
                cl = cls[i]
                s0[cl], s1[cl] = -s0[cl], -s1[cl]

    def getClusterCouplings(self, clusters, s0):
        """
        Get the effective cluster couplings as prescribed
        by Swendsen and Wang.
        input:
            clusters: list of lists of cluster indicies
            s0: reference spin configuration            
        returns:
            sparse matrix of effective couplings
        """
        bdrys = [boundaries(cls, self.bonds) for cls in clusters]
        bondmap = self.bondmap
        rows, cols, data = [], [], []
        for i, bdry_i in enumerate(bdrys[:-1]):
            for jj, bdry_j in enumerate(bdrys[i+1:]):
                shared = bdry_i.intersection(bdry_j)
                if shared:
                    coupling = sum([s0[bd[0]]*s0[bd[1]]*bondmap[bd] for bd in shared])
                    rows += [i, jj+i+1]
                    cols += [jj+i+1, i] #symmetric matrix
                    data += [coupling, coupling]
        N_c = len(clusters)
        return sps.csr_matrix((data, (rows, cols)), shape=(N_c, N_c))

    def __next__(self):
        """ For the iterator, do one monte carlo step"""
        for nt, T in enumerate(self.templist):
            for i, rand in enumerate(self._my_rng.rand(self.N)):
                flipFast(i, self.spinarray[nt], T, rand,
                         self.neighborlist[i], self.neighborbonds[i])
        for nt in range(self.Ntemps-1):
            self.clusterStep(self.spinarray[nt], self.templist[nt],
                             self.spinarray[nt+1], self.templist[nt+1])
        self._increment_time()
        return self

    def notify_observers(self):
        """ Send state to observers """
        for obs in self.observers:
            obs.send(self)
   

#-----------------------------
# Utilities
#-----------------------------

def neighbors(ind, adj):
    """
    Given adjacency matrix, return the neighbors
    of node 'ind'
    """
    if type(adj) is sps.csr.csr_matrix:
        return set(adj[ind].indices)
    elif type(adj) is np.ndarray:
        conn = adj[ind]
        return set(np.arange(len(conn))[adj[ind]>0])

def boundaries(cls, bonds):
    return reduce(lambda s, t: s.symmetric_difference(t), 
                  [bonds[i] for i in cls])

def getClusters(taus, neighborsets):
    """
    input:
        taus : values of overlap between two spin configurations.
               Array of N spin-values.
        neighborsets: list ordered by spin index, whose
        elements are the spins neighboring the spin at that index.
    output:
        list of cluster indices split according to the adjacency
        matrix to find contiguous regions of 'up' or 'down' spins
    """
    result = []
    spins = set(range(len(taus)))
    while spins:
        seed = spins.pop()
        newclust = {seed}
        queue = [seed]
        while queue:
            nxt = queue.pop(0)
            neighs = neighborsets[nxt].difference(newclust)
            for neigh in list(neighs):
                if not taus[neigh] == taus[seed]: neighs.remove(neigh)
            newclust.update(neighs)
            queue.extend(neighs)
            spins.difference_update(neighs) 
        result.append(list(newclust))
    return result

@jit("void(int64, int64[:], float64, float64, int32[:], float64[:])", 
     nopython=True)
def flipFast(index, s, T, rand, neighs, bonds):
    """
    Much more efficient version of _oneSweep MC
    input:
        index: spin to attempt flipping
        s: spin configuration
        T: temperature
        rand: a uniform random float
        neighs: list of neighboring sites to index
        bonds: list of bonds connections index to its neighbors
    """
    dE = 0.
    for nn, jj in zip(neighs, bonds):
        dE += s[nn]*jj
    dE *= 2*s[index]
    if dE < 0 or rand <= exp(-dE/T): s[index] *= -1

def coroutine(func):
    """ Decorator that initializes coroutines (return yield)"""
    @wraps(func)
    def warmed_up(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.send(None)
        return cr
    return warmed_up

def fileName(model, directory = 'time_series/', protocol = '.pkl',  appendnote = ''):
    """Creates a unique file name for a given model final string first 10
    characters from the current git commit hash"""
    hashstr = str(check_output("git log -n 1 | grep commit | sed s/commit\ //", shell=True)[:10])
    strformat = 'EA-seed_{seed}-size_{size}-T_{T}-' + appendnote + hashstr
    return directory + strformat.format(**model.__dict__) + protocol

def pickleLoader(pklfile):
    try:
        while True:
            yield pickle.load(pklfile)
    except EOFError:
        pass

def timeSeriesLoader(filename):
    series = []
    with open(filename, 'rb') as infile:
        for state in pickleLoader(infile):
            series += [state]
    return np.array(series)
        
def packSpins(spins):
    return np.packbits((spins+1)/2)

def unpackSpins(ints, N):
    return (2*np.unpackbits(ints).astype('int')-1)[:N]

#----------------------------
# Observers
#----------------------------

@coroutine
def saveTimeSeries(period = 1, directory = 'time_series/', protocol = '.npy', **kwargs):
    """Observer which saves spin state of model to a file. specify directory to save in,
    protocol: 'pkl' or 'h5 for pickling or hdf5"""
    #Make file upon recieving model with fileName()
    model = yield
    if protocol == '.pkl':
        name = fileName(model, directory, protocol, **kwargs)
        if 'filename' in kwargs:
            name = kwargs['filename']
        while True:
            model = yield
            if model.time % period == 0:
                with open(name,'ab') as outfile:
                    pickle.dump(model.spins.copy(), outfile,-1)
    elif protocol == '.npy':
        with open(name, 'wb') as outfile:
            while True:
                model = yield
                if model.time % period == 0:
                    np.save(outfile, packSpins(model.spins))
                    outfile.flush()

@coroutine
def onlinePairCorr(period=100, **kwargs):
    """Calculate the pair correlation function using the same online
    Welford-type algorithm as for the FIM. Only calculates cross moments.
    PairCorr is calculated upon pickling by dividing by self.pairtime and T**2.
    """
    model = yield
    model.pairtime = 0
    N = model.N
    model.meanSpinPair = np.zeros(N)
    model.cumSpinPair = np.zeros((N, N))
    while True:
        model = yield
        if model.time % period==0:
            model.pairtime += 1
            meanSpin_old = model.meanSpinPair.copy().astype('float')
            spins = model.spins.copy().astype('float')
            model.meanSpinPair += (spins - meanSpin_old)/(model.pairtime+0.0)
            model.cumSpinPair += np.outer(spins-model.meanSpinPair,
                                          spins-meanSpin_old)

#----------------------------
#   SuperCluster Calculations 
#----------------------------

def simplifyNestedClustering(supercluster, cluster):
    """
    Unpacks a clustering of clusters. supercluster is a list of indices from
    cluster, cluster is a list of original indicies. The output is the indices
    of cluster organizes according to supercluster
    """
    flattened = []
    for scl in supercluster:
        flat_super = []
        for cl in scl:
            flat_super += cluster[cl]
        flattened += [flat_super]
    return flattened

def mapCij(clust, CIJ):
    """
    Takes a ClusterModel correlation function and, using the cluster assignment
    inherited from an REMC model, maps the correlation function (CIJ) onto
    the dimensions of the original system correlations
    """
    N = np.sum(list(map(len, clust)))
    mapped_CIJ = np.zeros((N,N))
    for cl1, row in zip(clust, CIJ):
        for cl2, corr in zip(clust, row):
            for i_x in cl1:
                for i_y in cl2:
                    mapped_CIJ[i_x,i_y] = corr
    return mapped_CIJ

    





