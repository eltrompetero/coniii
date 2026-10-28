"""Nopython (@njit) Metropolis samplers for pairwise Ising and 3-state Potts models.

The chains match the Boost C++ samplers (cpp/samplers.cpp): one single-site
update attempt per iteration, an O(n) local energy difference, random or
systematic site order, and for Potts3 a proposal to one of the other two states
chosen uniformly. Output differs from the C++ for the same seed because numba
uses its own random number generator.

To match the pure-Python samplers, a chain can start from a given state, and
returns its final state (for chain continuation), the energy of each sample, and
optionally the energy after every step. Energies follow coniii's convention,
E = -theta . observables.

Seeding follows the Boost paths: the caller draws seeds from its own
RandomState and passes them in, so seeded runs are reproducible.

Workers in the parallel helpers are module-level functions. multiprocess
pickles them by reference, so each worker imports this module and gets the
cached compiled kernels. Passing a jitted function to a worker by value (in a
closure or as an attribute of a pickled object) would make numba recompile it
in every spawned worker.
"""
import multiprocess as mp
import numpy as np
from numba import njit


def coupling_matrix(n, J):
    """Dense symmetric coupling matrix from upper-triangle couplings (row-major, as in
    the C++ and in Metropolis.theta)."""
    Jmat = np.zeros((n, n))
    Jmat[np.triu_indices(n, 1)] = J
    return Jmat + Jmat.T


@njit(cache=True)
def seed(s):
    """Seed numba's generator. It is separate from NumPy's and can only be seeded from
    inside a jitted function."""
    np.random.seed(s)


# ============================================================================ #
# Random starting states and energies
# ============================================================================ #
@njit(cache=True)
def _random_ising_state(n):
    s = np.empty(n, dtype=np.int64)
    for i in range(n):
        s[i] = -1 if np.random.randint(0, 2) else 1
    return s


@njit(cache=True)
def _random_potts3_state(n):
    s = np.empty(n, dtype=np.int64)
    for i in range(n):
        s[i] = np.random.randint(0, 3)
    return s


@njit(cache=True)
def _ising_energy(h, Jmat, s):
    n = s.size
    E = 0.0
    for i in range(n):
        E -= h[i] * s[i]
        for j in range(i + 1, n):
            E -= Jmat[i, j] * s[i] * s[j]
    return E


@njit(cache=True)
def _potts3_energy(h3, Jmat, s):
    n = s.size
    E = 0.0
    for i in range(n):
        E -= h3[s[i], i]
        for j in range(i + 1, n):
            if s[i] == s[j]:
                E -= Jmat[i, j]
    return E


# ============================================================================ #
# Kernels
#
# s is the starting state and is updated in place, so it holds the final state on
# return. Returns (samples, energy of each sample, energy history). The history has
# the energy after every step past burn-in and the same length as the pure-Python
# path's (sample_size*n_iters + 1, last entry unused); it is empty unless
# save_history is True.
#
# The random number calls are in the same order as before starting states and
# energies were added, so seeded output is unchanged.
# ============================================================================ #
@njit(cache=True)
def _ising_run(h, Jmat, s, sample_size, burn_in, n_iters, systematic_iter, save_history):
    n = h.size
    E = _ising_energy(h, Jmat, s)
    out = np.empty((sample_size, n), dtype=np.int64)
    E_out = np.empty(sample_size)
    history = np.zeros(sample_size * n_iters + 1 if save_history else 0)
    counter = 0
    for step in range(burn_in + sample_size * n_iters):
        i = counter % n if systematic_iter else np.random.randint(0, n)
        counter += 1
        field = h[i]
        for j in range(n):
            field += Jmat[i, j] * s[j]
        de = 2.0 * s[i] * field
        if de <= 0.0 or np.random.random() <= np.exp(-de):
            s[i] = -s[i]
            E += de
        if step >= burn_in:
            if save_history:
                history[step - burn_in] = E
            if n_iters > 0 and (step - burn_in + 1) % n_iters == 0:
                k = (step - burn_in) // n_iters
                out[k] = s
                E_out[k] = E
    if n_iters == 0:
        for k in range(sample_size):
            out[k] = s
            E_out[k] = E
    return out, E_out, history


@njit(cache=True)
def _potts3_run(h3, Jmat, s, sample_size, burn_in, n_iters, systematic_iter, save_history):
    n = h3.shape[1]
    E = _potts3_energy(h3, Jmat, s)
    out = np.empty((sample_size, n), dtype=np.int64)
    E_out = np.empty(sample_size)
    history = np.zeros(sample_size * n_iters + 1 if save_history else 0)
    counter = 0
    for step in range(burn_in + sample_size * n_iters):
        i = counter % n if systematic_iter else np.random.randint(0, n)
        counter += 1
        a = s[i]
        b = (a + np.random.randint(1, 3)) % 3
        de = h3[a, i] - h3[b, i]
        for j in range(n):          # J[i, i] == 0, so j == i contributes nothing
            if s[j] == a:
                de += Jmat[i, j]
            if s[j] == b:
                de -= Jmat[i, j]
        if de <= 0.0 or np.random.random() <= np.exp(-de):
            s[i] = b
            E += de
        if step >= burn_in:
            if save_history:
                history[step - burn_in] = E
            if n_iters > 0 and (step - burn_in + 1) % n_iters == 0:
                k = (step - burn_in) // n_iters
                out[k] = s
                E_out[k] = E
    if n_iters == 0:
        for k in range(sample_size):
            out[k] = s
            E_out[k] = E
    return out, E_out, history


# ============================================================================ #
# Python-level entry points
# ============================================================================ #
def _ising_args(n, theta):
    theta = np.asarray(theta, dtype=np.float64)
    return theta[:n].copy(), coupling_matrix(n, theta[n:])


def _potts3_args(n, theta):
    theta = np.asarray(theta, dtype=np.float64)
    return theta[:3*n].reshape(3, n).copy(), coupling_matrix(n, theta[3*n:])


def _start(n, initial_state, random_state):
    if initial_state is None:
        return random_state(n)
    s = np.array(initial_state, dtype=np.int64).ravel()
    assert s.size == n, "initial_state must have n elements."
    return s


def ising_chain(n, theta, sample_size, burn_in, n_iters, systematic_iter=False,
                initial_state=None, save_history=False):
    """Run one Ising chain. Seed with seed() first.

    Parameters
    ----------
    initial_state : ndarray, None
        Starting state with values +/-1. If None, a random state is drawn from numba's
        generator.
    save_history : bool, False
        If True, also return the energy after every step past burn-in.

    Returns
    -------
    sample : ndarray of shape (sample_size, n)
    E : ndarray of shape (sample_size,)
        Energy of each sample.
    final_state : ndarray of shape (n,)
    history : ndarray
        Empty unless save_history is True.
    """
    h, Jmat = _ising_args(n, theta)
    s = _start(n, initial_state, _random_ising_state)
    out, E, history = _ising_run(h, Jmat, s, sample_size, burn_in, n_iters,
                                 systematic_iter, save_history)
    return out, E, s, history


def potts3_chain(n, theta, sample_size, burn_in, n_iters, systematic_iter=False,
                 initial_state=None, save_history=False):
    """Potts3 counterpart of ising_chain; states take values 0, 1, 2."""
    h3, Jmat = _potts3_args(n, theta)
    s = _start(n, initial_state, _random_potts3_state)
    out, E, history = _potts3_run(h3, Jmat, s, sample_size, burn_in, n_iters,
                                  systematic_iter, save_history)
    return out, E, s, history


def ising_sample(n, theta, sample_size, burn_in, n_iters, systematic_iter=False):
    """Ising sample of shape (sample_size, n) from a random start. Seed with seed()
    first."""
    return ising_chain(n, theta, sample_size, burn_in, n_iters, systematic_iter)[0]


def potts3_sample(n, theta, sample_size, burn_in, n_iters, systematic_iter=False):
    """Potts3 sample of shape (sample_size, n) from a random start. Seed with seed()
    first."""
    return potts3_chain(n, theta, sample_size, burn_in, n_iters, systematic_iter)[0]


# ============================================================================ #
# Parallel
# ============================================================================ #
def _ising_worker(args):
    h, Jmat, size, burn_in, n_iters, systematic_iter, sd, s0 = args
    seed(int(sd))   # int(): randint gives int32 on Windows; keep one compiled signature
    s = _random_ising_state(h.size) if s0 is None else s0.copy()
    out, E, _ = _ising_run(h, Jmat, s, size, burn_in, n_iters, systematic_iter, False)
    return out, E, s


def _potts3_worker(args):
    h3, Jmat, size, burn_in, n_iters, systematic_iter, sd, s0 = args
    seed(int(sd))
    s = _random_potts3_state(h3.shape[1]) if s0 is None else s0.copy()
    out, E, _ = _potts3_run(h3, Jmat, s, size, burn_in, n_iters, systematic_iter, False)
    return out, E, s


def _chain_parallel(worker, kernel_args, n, sample_size, burn_in, n_iters,
                    systematic_iter, seeds, initial_states):
    n_chains = len(seeds)
    if initial_states is None:
        starts = [None] * n_chains
    else:
        starts = np.array(initial_states, dtype=np.int64)
        assert starts.shape == (n_chains, n), "initial_states must have shape (len(seeds), n)."
    size = int(np.ceil(sample_size / n_chains))
    jobs = [kernel_args + (size, burn_in, n_iters, systematic_iter, sd, s0)
            for sd, s0 in zip(seeds, starts)]
    with mp.Pool(n_chains) as pool:
        out, E, final = zip(*pool.map(worker, jobs))
    # each chain's final state is its last sample, so its final energy is its last energy
    final_E = np.array([e[-1] for e in E])
    return (np.vstack(out)[:sample_size], np.concatenate(E)[:sample_size],
            np.vstack(final), final_E)


def ising_chain_parallel(n, theta, sample_size, burn_in, n_iters, systematic_iter,
                         seeds, initial_states=None):
    """Run one Ising chain per seed in a pool of len(seeds) workers.

    Each chain returns ceil(sample_size / len(seeds)) samples, and the stacked samples
    and energies are truncated to sample_size.

    Parameters
    ----------
    seeds : sequence of int
        One seed per chain.
    initial_states : ndarray, None
        Starting states of shape (len(seeds), n). If None, each chain starts from a
        random state drawn after seeding.

    Returns
    -------
    sample : ndarray of shape (sample_size, n)
    E : ndarray of shape (sample_size,)
        Energy of each sample.
    final_states : ndarray of shape (len(seeds), n)
    final_E : ndarray of shape (len(seeds),)
        Energy of each chain's final state.
    """
    return _chain_parallel(_ising_worker, _ising_args(n, theta), n, sample_size,
                           burn_in, n_iters, systematic_iter, seeds, initial_states)


def potts3_chain_parallel(n, theta, sample_size, burn_in, n_iters, systematic_iter,
                          seeds, initial_states=None):
    """Potts3 counterpart of ising_chain_parallel."""
    return _chain_parallel(_potts3_worker, _potts3_args(n, theta), n, sample_size,
                           burn_in, n_iters, systematic_iter, seeds, initial_states)


def ising_sample_parallel(n, theta, sample_size, burn_in, n_iters, systematic_iter,
                          seeds):
    """Samples only, from random starts. See ising_chain_parallel."""
    return ising_chain_parallel(n, theta, sample_size, burn_in, n_iters,
                                systematic_iter, seeds)[0]


def potts3_sample_parallel(n, theta, sample_size, burn_in, n_iters, systematic_iter,
                           seeds):
    """Samples only, from random starts. See potts3_chain_parallel."""
    return potts3_chain_parallel(n, theta, sample_size, burn_in, n_iters,
                                 systematic_iter, seeds)[0]