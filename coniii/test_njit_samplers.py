"""Tests for coniii.njit_samplers.

Statistical check: at n = 5, sampled magnetizations and pair statistics are compared
with exact values from enumerating all 32 (Ising) or 243 (Potts3) states, as z-scores
against the exact standard errors. Samples are 10 sweeps apart, which is effectively
independent at n = 5, so a correct sampler's max |z| is usually 1.5-3. The test fails
if it exceeds 4, which happens by chance in well under 1 run in 500. This runs on
every platform and numba version.

Fingerprint check: numba's np.random is its own Mersenne Twister, independent of the
platform's C library, so for a given numba version the seeded output should be
bit-identical on Linux, macOS and Windows. FINGERPRINTS records the expected output per
numba version; the exact check is skipped for versions not in the table. To add a
version, run this file directly and paste the printed entry into FINGERPRINTS.
"""
import hashlib
import itertools

import numba
import numpy as np
import pytest

from coniii.njit_samplers import (seed, ising_sample, potts3_sample,
                                  ising_sample_parallel, potts3_sample_parallel,
                                  ising_chain, potts3_chain,
                                  ising_chain_parallel, potts3_chain_parallel)

N, BURN, SEED = 5, 1000, 1
SAMPLES, ITERS = 20000, 10 * N      # 10 sweeps between samples
Z_FAIL = 4.0
IU = np.triu_indices(N, 1)

# numba version -> {model: first 16 hex digits of sha256 of the seeded int64 output}
FINGERPRINTS = {
    '0.67.0': {
        'Ising': '4531993884076d35',
        'Potts3': '61af6168486959dc',
    },
    '0.68.1': {
        'Ising': '4531993884076d35',
        'Potts3': '61af6168486959dc',
    },
}


def ising_feats(S):
    return np.hstack((S, S[:, IU[0]] * S[:, IU[1]])).astype(float)


def potts3_feats(S):
    return np.hstack([S == k for k in range(3)] + [S[:, IU[0]] == S[:, IU[1]]]).astype(float)


MODELS = {  # observables, number of field parameters, sampler, values, all states
    "Ising": (ising_feats, N, ising_sample, {-1, 1},
              np.array(list(itertools.product([-1, 1], repeat=N)))),
    "Potts3": (potts3_feats, 3 * N, potts3_sample, {0, 1, 2},
               np.array(list(itertools.product(range(3), repeat=N)))),
}
PARALLEL = {"Ising": ising_sample_parallel, "Potts3": potts3_sample_parallel}
CHAIN = {"Ising": ising_chain, "Potts3": potts3_chain}
CHAIN_PARALLEL = {"Ising": ising_chain_parallel, "Potts3": potts3_chain_parallel}


def _theta(model):
    n_fields = MODELS[model][1]
    return np.random.RandomState(0).normal(scale=0.5, size=n_fields + N*(N-1)//2)


def _draw(model, systematic):
    seed(SEED)
    return MODELS[model][2](N, _theta(model), SAMPLES, BURN, ITERS, systematic)


def _fingerprint(X):
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.int64).tobytes()).hexdigest()[:16]


def exact_moments(feats, states, theta):
    """Exact mean and SD of each observable. Both models have energy -theta . observables."""
    F = feats(states)
    E = -F @ theta
    p = np.exp(-(E - E.min()))
    p /= p.sum()
    mu = p @ F
    return mu, np.sqrt(p @ F**2 - mu**2)


@pytest.mark.parametrize("systematic", [False, True])
@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_exact_moments(model, systematic):
    feats, _, _, values, states = MODELS[model]
    X = _draw(model, systematic)
    assert X.shape == (SAMPLES, N)
    assert set(np.unique(X).tolist()) <= values

    mu, sd = exact_moments(feats, states, _theta(model))
    z = np.abs((feats(X).mean(0) - mu) / (sd / np.sqrt(SAMPLES))).max()
    assert z <= Z_FAIL, f"{model} systematic={systematic}: max|z| = {z:.2f}"


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_fingerprint(model):
    if numba.__version__ not in FINGERPRINTS:
        pytest.skip(f"no fingerprint recorded for numba {numba.__version__}")
    assert _fingerprint(_draw(model, False)) == FINGERPRINTS[numba.__version__][model]


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_seeded_reproducible(model):
    assert np.array_equal(_draw(model, False), _draw(model, False))


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_parallel(model):
    values, sampler = MODELS[model][3], MODELS[model][2]
    theta = _theta(model)
    seeds = np.random.RandomState(0).randint(2**31 - 1, size=2)

    X1 = PARALLEL[model](N, theta, 101, 20, 20, False, seeds)
    X2 = PARALLEL[model](N, theta, 101, 20, 20, False, seeds)
    assert X1.shape == (101, N)
    assert set(np.unique(X1).tolist()) <= values
    assert np.array_equal(X1, X2)

    # each worker runs the serial kernel: chain 0 matches a serial run with seeds[0]
    seed(int(seeds[0]))
    assert np.array_equal(X1[:51], sampler(N, theta, 51, 20, 20, False))


def _energies(model, X):
    """Energies computed directly from the samples: E = -theta . observables."""
    return -MODELS[model][0](X) @ _theta(model)


@pytest.mark.parametrize("systematic", [False, True])
@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_chain_energies_history(model, systematic):
    sample_size, n_iters = 200, 7
    seed(SEED)
    X, E, final, history = CHAIN[model](N, _theta(model), sample_size, 50, n_iters,
                                        systematic, save_history=True)
    assert np.allclose(E, _energies(model, X))
    assert np.array_equal(final, X[-1])
    # history has the pure-Python path's length; each sample is taken after the last
    # step of its block
    assert history.size == sample_size * n_iters + 1
    assert np.allclose(history[n_iters-1:-1:n_iters], E)

    seed(SEED)
    assert CHAIN[model](N, _theta(model), sample_size, 50, n_iters, systematic)[3].size == 0


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_chain_continuation(model):
    # With numba's generator seeded once, 100 samples then 100 more from the final state
    # (no burn-in) reproduce one run of 200. Random site order: the systematic counter
    # restarts at each call, so it only lines up when the step counts happen to.
    theta = _theta(model)
    seed(SEED)
    full = CHAIN[model](N, theta, 200, 50, 7)[0]
    seed(SEED)
    first, _, final, _ = CHAIN[model](N, theta, 100, 50, 7)
    second = CHAIN[model](N, theta, 100, 0, 7, initial_state=final)[0]
    assert np.array_equal(np.vstack((first, second)), full)


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_chain_initial_state(model):
    # with no steps, every sample is the starting state
    s0 = np.array([1, -1, 1, -1, 1]) if model == "Ising" else np.array([0, 1, 2, 0, 1])
    seed(SEED)
    X, E, final, _ = CHAIN[model](N, _theta(model), 3, 0, 0, initial_state=s0)
    assert np.array_equal(X, np.tile(s0, (3, 1)))
    assert np.array_equal(final, s0)
    assert np.allclose(E, _energies(model, X))


@pytest.mark.parametrize("model", ["Ising", "Potts3"])
def test_chain_parallel(model):
    values = MODELS[model][3]
    theta = _theta(model)
    seeds = np.random.RandomState(0).randint(2**31 - 1, size=2)

    X, E, final, final_E = CHAIN_PARALLEL[model](N, theta, 101, 20, 20, False, seeds)
    assert X.shape == (101, N) and E.shape == (101,) and final.shape == (2, N)
    assert set(np.unique(final).tolist()) <= values
    assert np.allclose(E, _energies(model, X))
    assert final_E.shape == (2,)
    assert np.allclose(final_E, _energies(model, final))
    # chain 1 holds rows 51..101 of its 51 samples; its final state is its last sample,
    # which truncation to 101 rows drops, so compare chain 0 only
    assert np.array_equal(final[0], X[50])

    # starting states: one per chain, and continuation from the returned final states
    X2, E2, final2, _ = CHAIN_PARALLEL[model](N, theta, 101, 0, 20, False, seeds,
                                              initial_states=final)
    assert final2.shape == (2, N)
    assert np.allclose(E2, _energies(model, X2))
    with pytest.raises(AssertionError):
        CHAIN_PARALLEL[model](N, theta, 101, 0, 20, False, seeds,
                              initial_states=final[:1])


if __name__ == "__main__":
    print(f"    {numba.__version__!r}: {{")
    for model in MODELS:
        print(f"        {model!r}: {_fingerprint(_draw(model, False))!r},")
    print("    },")