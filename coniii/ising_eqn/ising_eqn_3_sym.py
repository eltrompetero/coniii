# Equations for 3-spin Ising model.

# Written on 2018/12/12.
from numpy import zeros, exp, array, prod, isnan
from scipy.special import logsumexp

def calc_observables(params):
    """
        Give each set of parameters concatenated into one array.
        """
    Cout = zeros((6))
    H = params[0:3]
    J = params[3:6]
    energyTerms = [    +H[0]+H[1]+H[2]+J[0]+J[1]+J[2], +H[0]+H[1]-H[2]+J[0]-J[1]-J[2], +H[0]-H[1]+H[2]-J[0]+J[1]-J[2], +H[0]-H[1]-H[2]-J[0]-J[1]+
    J[2], -H[0]+H[1]+H[2]-J[0]-J[1]+J[2], -H[0]+H[1]-H[2]-J[0]+J[1]-J[2], -H[0]-H[1]+H[2]+J[0]-J[1]-J[2], -H[0]-H[1]-H[2]+
            J[0]+J[1]+J[2],]
    logZ = logsumexp(energyTerms)
    num = logsumexp(energyTerms, b=[ 1, 1, 1, 1,-1,-1,-1,-1], return_sign=True)
    Cout[0] = exp( num[0] - logZ ) * num[1]
    num = logsumexp(energyTerms, b=[ 1, 1,-1,-1, 1, 1,-1,-1], return_sign=True)
    Cout[1] = exp( num[0] - logZ ) * num[1]
    num = logsumexp(energyTerms, b=[ 1,-1, 1,-1, 1,-1, 1,-1], return_sign=True)
    Cout[2] = exp( num[0] - logZ ) * num[1]
    num = logsumexp(energyTerms, b=[ 1, 1,-1,-1,-1,-1, 1, 1], return_sign=True)
    Cout[3] = exp( num[0] - logZ ) * num[1]
    num = logsumexp(energyTerms, b=[ 1,-1, 1,-1,-1, 1,-1, 1], return_sign=True)
    Cout[4] = exp( num[0] - logZ ) * num[1]
    num = logsumexp(energyTerms, b=[ 1,-1,-1, 1, 1,-1,-1, 1], return_sign=True)
    Cout[5] = exp( num[0] - logZ ) * num[1]
    Cout[isnan(Cout)] = 0.
    return(Cout)

def p(params):
    """
        Give each set of parameters concatenated into one array.
        """
    Cout = zeros((6))
    H = params[0:3]
    J = params[3:6]
    H = params[0:3]
    J = params[3:6]
    Pout = zeros((8))
    energyTerms = [    +H[0]+H[1]+H[2]+J[0]+J[1]+J[2], +H[0]+H[1]-H[2]+J[0]-J[1]-J[2], +H[0]-H[1]+H[2]-J[0]+J[1]-J[2], +H[0]-H[1]-H[2]-J[0]-J[1]+
    J[2], -H[0]+H[1]+H[2]-J[0]-J[1]+J[2], -H[0]+H[1]-H[2]-J[0]+J[1]-J[2], -H[0]-H[1]+H[2]+J[0]-J[1]-J[2], -H[0]-H[1]-H[2]+
            J[0]+J[1]+J[2],]
    logZ = logsumexp(energyTerms)
    Pout[0] = exp( +H[0]+H[1]+H[2]+J[0]+J[1]+J[2] - logZ )
    Pout[1] = exp( +H[0]+H[1]-H[2]+J[0]-J[1]-J[2] - logZ )
    Pout[2] = exp( +H[0]-H[1]+H[2]-J[0]+J[1]-J[2] - logZ )
    Pout[3] = exp( +H[0]-H[1]-H[2]-J[0]-J[1]+J[2] - logZ )
    Pout[4] = exp( -H[0]+H[1]+H[2]-J[0]-J[1]+J[2] - logZ )
    Pout[5] = exp( -H[0]+H[1]-H[2]-J[0]+J[1]-J[2] - logZ )
    Pout[6] = exp( -H[0]-H[1]+H[2]+J[0]-J[1]-J[2] - logZ )
    Pout[7] = exp( -H[0]-H[1]-H[2]+J[0]+J[1]+J[2] - logZ )

    Pout = Pout[::-1]
    return(Pout)
