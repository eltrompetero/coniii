"""Seeded reference outputs for the Boost C++ samplers.

Run this file directly, in an environment where the Boost extension is built,
to regenerate coniii/test_data/boost_reference.npz. The test compares fresh
output against that file, so any change to the C++ sampling path (e.g. a
pybind11 port) must reproduce it exactly on the same platform.

Potts3 is covered through the parallel path only: with the extension built,
serial Potts3 currently samples via BoostIsing (#42).
"""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from coniii import samplers
from coniii.samplers import Metropolis, Potts3

REF = Path(__file__).parent / "test_data" / "boost_reference.npz"

CASES = [(5, 0), (10, 1)]    # (n, seed)
SAMPLE_SIZE, N_ITERS, BURN_IN = 200, 20, 50
N_CPUS = 2                   # parallel output depends on cpu_count; pin it


def _theta(n, seed):
    # legacy RandomState is stable across platforms and NumPy versions
    return np.random.RandomState(seed).normal(scale=0.3, size=n*(n-1)//2 + n)


def _theta_potts3(n, seed):
    return np.random.RandomState(seed).normal(scale=0.3, size=3*n + n*(n-1)//2)


def generate():
    assert samplers.IMPORTED_SAMPLERS_EXT, "Boost extension not available"
    kw = dict(n_iters=N_ITERS, burn_in=BURN_IN)
    out = {}
    with mock.patch.object(samplers.mp, "cpu_count", return_value=N_CPUS):
        for n, seed in CASES:
            theta = _theta(n, seed)
            theta3 = _theta_potts3(n, seed)
            for sys_iter in (False, True):
                key = f"n{n}_sys{int(sys_iter)}"

                # two consecutive serial calls: checks the rng advances correctly
                m = Metropolis(n, theta, rng=np.random.RandomState(seed), iprint=False)
                m.generate_sample(SAMPLE_SIZE, systematic_iter=sys_iter, **kw)
                out[f"{key}_serial1"] = m.sample.astype(np.int8)
                m.generate_sample(SAMPLE_SIZE, systematic_iter=sys_iter, **kw)
                out[f"{key}_serial2"] = m.sample.astype(np.int8)

                m = Metropolis(n, theta, rng=np.random.RandomState(seed), iprint=False)
                m.generate_sample_parallel(SAMPLE_SIZE, systematic_iter=sys_iter, **kw)
                out[f"{key}_parallel"] = m.sample.astype(np.int8)

                # Potts3 parallel only (serial is #42); uses n_cpus, not cpu_count
                p = Potts3(n, theta3, n_cpus=N_CPUS, rng=np.random.RandomState(seed))
                p.generate_sample_parallel(SAMPLE_SIZE, systematic_iter=sys_iter, **kw)
                out[f"{key}_potts3_parallel"] = p.sample.astype(np.int8)
    return out


@pytest.mark.skipif(not samplers.IMPORTED_SAMPLERS_EXT,
                    reason="Boost extension not built")
def test_boost_reference():
    ref = np.load(REF)
    new = generate()
    assert set(ref.files) == set(new)
    for k in ref.files:
        same = np.array_equal(ref[k], new[k])
        assert same, f"{k} differs from reference ({np.sum(ref[k] != new[k])} entries)"


if __name__ == "__main__":
    REF.parent.mkdir(exist_ok=True)
    np.savez_compressed(REF, **generate())
    print(f"wrote {REF}")