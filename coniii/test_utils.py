# =============================================================================================== #
# Testing for utils.py module.
# Released with ConIII package.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
from .utils import *
import sys
version = sys.version_info
assert version.major>=3 and version.minor>=6


def test_pair_corr():
    from itertools import combinations
    np.random.seed(0)

    X = np.random.choice([-1,1], size=(5,3))
    si_ = X.mean(0)
    sisj_ = np.array([(X[:,i]*X[:,j]).mean(0) for i,j in combinations(range(3),2)])

    si, sisj = pair_corr(X)
    assert np.isclose(si,si_).all()
    assert np.isclose(sisj, sisj_).all()

    si, sisj = pair_corr(X, weights=1/len(X))
    assert np.isclose(si,si_).all()
    assert np.isclose(sisj, sisj_).all()

    si, sisj = pair_corr(X, weights=np.zeros(len(X))+1/len(X))
    assert np.isclose(si, si_).all()
    assert np.isclose(sisj, sisj_).all()
    
    # try exclude_empty switch
    X = np.vstack((X, np.zeros(3)))
    si, sisj = pair_corr(X, exclude_empty=True)
    assert np.isclose(si, si_).all()
    assert np.isclose(sisj, sisj_).all()

    si, sisj = pair_corr(X, exclude_empty=True, laplace_count=True)
    assert np.isclose(si, si_*5/7).all()
    assert np.isclose(sisj, sisj_*5/9).all()

def test_sub_to_ind():
    for n in range(2,5):
        counter = 0
        for i,j in combinations(list(range(n)),2):
            assert sub_to_ind(n,i,j)==counter
            assert ind_to_sub(n,counter)==(i,j)
            counter += 1

def test_state_gen_and_count():
    """Test generation of binary states using bin_states() and xbin_states()."""
    assert np.array_equal( bin_states(5), np.vstack([i for i in xbin_states(5)]) )

    states = bin_states(5)
    p, s = state_probs(states)
    assert np.isclose(p, 1/32).all()
    assert np.array_equal(s, states)

    states = bin_states(5, sym=True)
    p, s = state_probs(states)
    assert np.isclose(p, 1/32).all()
    assert np.array_equal(s[::-1], states)

def test_convert_corr():
    np.random.seed(0)
    X = np.random.choice([-1,1], size=(100,3))
    
    # test conversion from 11 to 01 basis
    sisj11to01 = convert_corr(*pair_corr(X), '01', concat=True)
    sisj01 = pair_corr((X+1)/2, concat=True)
    assert np.isclose(sisj11to01, sisj01).all()
    
    # test conversion from 01 to 11 basis
    sisj01to11 = convert_corr(sisj01[:3], sisj01[3:], '11', concat=True)
    sisj11 = pair_corr(X, concat=True)
    assert np.isclose(sisj01to11, sisj11).all()

def test_convert_params():
    from .utils import _expand_binomial
    np.random.seed(0)

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
  
def test_define_ising_helper_functions():
    from scipy.spatial.distance import squareform
    from .ising_eqn import ising_eqn_3_sym as ising
    calc_e, calc_observables, mch_approximation = define_ising_helper_functions()

    np.random.seed(0)
    X = np.random.choice([-1,1],size=(10,3))
    h = np.random.normal(size=3)
    J = np.random.normal(size=3)
    hJ = np.concatenate((h, J))
    
    # check that calculation of energy is correct
    assert np.isclose( calc_e(X, hJ),
                       -X.dot(h)-(X.dot(squareform(J)).dot(X.T)).diagonal()/2 ).all()
    assert np.isclose( calc_e(X, hJ ),
                       -calc_observables(X).dot(hJ) ).all()
    
    # check that mch_approximation doesn't change anything when dlambda=0
    sisj = calc_observables(X).mean(0)
    newsisj = mch_approximation(X, np.zeros(6))
    assert np.isclose(sisj, newsisj).all()

    # check that jacobian estimated using mch_approximation is close to numerical approximation
    # first sample states from distribution that will be used to estimate pairwise correlations
    eps = 1e-6
    p = ising.p(hJ)
    X = np.array(bin_states(3, sym=True))[np.random.choice(range(8), size=80000)]
    # estimate jacobian using known parameters and by using mch approximation method
    jac = np.zeros((6,6))
    jacTest = np.zeros((6,6))
    for i in range(6):
        hJ_ = hJ.copy()
        hJ__ = hJ.copy()
        hJ_[i] -= eps
        hJ__[i] += eps
        jac[i] = (mch_approximation(X, hJ__) - mch_approximation(X, hJ_))/2/eps
        jacTest[i] = (ising.calc_observables(hJ__) - ising.calc_observables(hJ_))/2/eps
    assert (np.abs(jac-jacTest)<3e-2).all()

def test_adj():
    np.random.seed(0)
    s = np.random.randint(2, size=10)
    neighbors = adj(s)
    assert neighbors.shape==(10,10)
    assert ((s!=neighbors).sum(1)==1).all()

    s = 2*s-1
    neighbors = adj_sym(s)
    assert neighbors.shape==(10,10)
    assert ((s!=neighbors).sum(1)==1).all()

def test_calc_de():
    np.random.seed(0)
    s = np.random.randint(2, size=10)*2-1
    for i in range(10):
        assert calc_de(s[None,:], i)==-s[i]
    counter = 10
    for i in range(9):
        for j in range(i+1,10):
            assert calc_de(s[None,:], counter)==-s[i]*s[j]
            counter+=1

def test_base_repr():
    # some misc test cases
    for i in range(50,80):
        for base in [2,3,5,18]:
            print('i=%d, base=%d'%(i, base))
            print('my algo '+''.join(base_repr(i,base)), '\nnumpy', np.base_repr(i,base))
            assert ''.join(base_repr(i,base))==np.base_repr(i,base)
    
    # systematically increasing power: can be a difficult case (given floating point rep)
    for i in range(2,37):
        for power in range(1,7):
            assert ''.join(base_repr(i**power,i))=='1'+'0'*power

def test_vec2mat():
    for n in range(3,100):
        x = np.random.rand(n+n*(n-1)//2)
        assert np.array_equal(x, mat2vec(vec2mat(x)))
        xvec, xmat = vec2mat(x, True)
        assert np.array_equal(xvec, x[:n])

if __name__=='__main__':
    test_convert_params()
