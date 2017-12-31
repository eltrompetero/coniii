from __future__ import division
from utils import *

def test_sub_to_ind():
    for n in xrange(2,5):
        counter = 0
        for i,j in combinations(range(n),2):
            assert sub_to_ind(n,i,j)==counter
            assert ind_to_sub(n,counter)==(i,j)
            counter += 1

def test_state_gen_and_count():
    """Test generation of binary states using bin_states() and xbin_states()."""
    assert np.array_equal( bin_states(5),np.vstack([i for i in xbin_states(5)]) )

    states = bin_states(5)
    p,s = state_probs(states)
    assert (p==1/32).all()
    assert np.array_equal(s,states)

    states = bin_states(5,sym=True)
    p,s = state_probs(states)
    assert (p==1/32).all()
    assert np.array_equal(s,states)

