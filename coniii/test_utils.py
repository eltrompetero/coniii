# =============================================================================================== #
# Testing for utils.py module.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
from .utils import *


def test_sub_to_ind():
    for n in range(2,5):
        counter = 0
        for i,j in combinations(list(range(n)),2):
            assert sub_to_ind(n,i,j)==counter
            assert ind_to_sub(n,counter)==(i,j)
            counter += 1

def test_state_gen_and_count():
    """Test generation of binary states using bin_states() and xbin_states()."""
    assert np.array_equal( bin_states(5),np.vstack([i for i in xbin_states(5)]) )

    states = bin_states(5)
    p, s = state_probs(states)
    assert np.isclose(p, 1/32).all()
    assert np.array_equal(s,states)

    states = bin_states(5, sym=True)
    p, s = state_probs(states)
    assert np.isclose(p, 1/32).all()
    assert np.array_equal(s, states)

def test_convert_params():
    from .utils import _expand_binomial
    terms = _expand_binomial(np.exp(1), np.pi, 2)
    assert len(terms)==4
    assert terms[0]==np.exp(1)**2 and terms[1]==np.exp(1)*np.pi and terms[3]==np.pi**2
    
    from itertools import combinations
    n=9
    # iterate through indices to several dimensions of tensors
    for d in range(2,5):
        for i,multidimix in enumerate(combinations(range(n),d)):
            assert i==unravel_index(multidimix,n), (i,multidimix,unravel_index(pairix,n))

    h = np.random.normal(size=n)
    J = np.random.normal(size=n*(n-1)//2)
    h1, J1 = convert_params(h, J, convert_to='01')
    h2, J2 = ising_convert_params([h,J], convert_to='01')
    assert np.isclose(h1, h2).all() and np.isclose(J1, J2).all()
    h1, J1 = convert_params(h, J, convert_to='11')
    h2, J2 = ising_convert_params([h,J], convert_to='11')
    assert np.isclose(h1, h2).all() and np.isclose(J1, J2).all()
   
if __name__=='__main__':
    test_convert_params()
