# MIT License
# 
# Copyright (c) 2017 Edward D. Lee, Bryan C. Daniels
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

# Equations for 4-spin Ising model.

# Written on 2018/12/17.
from numpy import zeros, exp, array, prod, isnan
from ..enumerate import fast_logsumexp

def calc_observables(params):
    """
    Give all parameters concatenated into one array from lowest to highest order.
    Returns all correlations.
    """
    Cout = zeros((10))
    H = params[0:4]
    J = params[4:10]
    energyTerms = array([    +H[0]+H[1]+H[2]+H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5], +H[0]+H[1]+H[2]-H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5], +
    H[0]+H[1]-H[2]+H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5], +H[0]+H[1]-H[2]-H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5], +
    H[0]-H[1]+H[2]+H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5], +H[0]-H[1]+H[2]-H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5], +
    H[0]-H[1]-H[2]+H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5], +H[0]-H[1]-H[2]-H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5], -H[0]+
    H[1]+H[2]+H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5], -H[0]+H[1]+H[2]-H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5], -H[0]+
    H[1]-H[2]+H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5], -H[0]+H[1]-H[2]-H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5], -H[0]-H[1]+
    H[2]+H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5], -H[0]-H[1]+H[2]-H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5], -H[0]-H[1]-H[2]+
            H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5], -H[0]-H[1]-H[2]-H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5],])
    logZ = fast_logsumexp(energyTerms)[0]
    num = fast_logsumexp(energyTerms, [ 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1])
    Cout[0] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1])
    Cout[1] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1])
    Cout[2] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])
    Cout[3] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1])
    Cout[4] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1])
    Cout[5] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1])
    Cout[6] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1])
    Cout[7] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1])
    Cout[8] = exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1])
    Cout[9] = exp( num[0] - logZ ) * num[1]
    Cout[isnan(Cout)] = 0.
    return(Cout)

def p(params):
    """
    Give all parameters concatenated into one array from lowest to highest order.
    Returns probabilities of all configurations.
    """
    Cout = zeros((10))
    H = params[0:4]
    J = params[4:10]
    H = params[0:4]
    J = params[4:10]
    Pout = zeros((16))
    energyTerms = array([    +H[0]+H[1]+H[2]+H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5], +H[0]+H[1]+H[2]-H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5], +
    H[0]+H[1]-H[2]+H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5], +H[0]+H[1]-H[2]-H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5], +
    H[0]-H[1]+H[2]+H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5], +H[0]-H[1]+H[2]-H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5], +
    H[0]-H[1]-H[2]+H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5], +H[0]-H[1]-H[2]-H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5], -H[0]+
    H[1]+H[2]+H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5], -H[0]+H[1]+H[2]-H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5], -H[0]+
    H[1]-H[2]+H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5], -H[0]+H[1]-H[2]-H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5], -H[0]-H[1]+
    H[2]+H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5], -H[0]-H[1]+H[2]-H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5], -H[0]-H[1]-H[2]+
            H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5], -H[0]-H[1]-H[2]-H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5],])
    logZ = fast_logsumexp(energyTerms)[0]
    Pout[0] = exp( +H[0]+H[1]+H[2]+H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5] - logZ )
    Pout[1] = exp( +H[0]+H[1]+H[2]-H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5] - logZ )
    Pout[2] = exp( +H[0]+H[1]-H[2]+H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5] - logZ )
    Pout[3] = exp( +H[0]+H[1]-H[2]-H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5] - logZ )
    Pout[4] = exp( +H[0]-H[1]+H[2]+H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5] - logZ )
    Pout[5] = exp( +H[0]-H[1]+H[2]-H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5] - logZ )
    Pout[6] = exp( +H[0]-H[1]-H[2]+H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5] - logZ )
    Pout[7] = exp( +H[0]-H[1]-H[2]-H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5] - logZ )
    Pout[8] = exp( -H[0]+H[1]+H[2]+H[3]-J[0]-J[1]-J[2]+J[3]+J[4]+J[5] - logZ )
    Pout[9] = exp( -H[0]+H[1]+H[2]-H[3]-J[0]-J[1]+J[2]+J[3]-J[4]-J[5] - logZ )
    Pout[10] = exp( -H[0]+H[1]-H[2]+H[3]-J[0]+J[1]-J[2]-J[3]+J[4]-J[5] - logZ )
    Pout[11] = exp( -H[0]+H[1]-H[2]-H[3]-J[0]+J[1]+J[2]-J[3]-J[4]+J[5] - logZ )
    Pout[12] = exp( -H[0]-H[1]+H[2]+H[3]+J[0]-J[1]-J[2]-J[3]-J[4]+J[5] - logZ )
    Pout[13] = exp( -H[0]-H[1]+H[2]-H[3]+J[0]-J[1]+J[2]-J[3]+J[4]-J[5] - logZ )
    Pout[14] = exp( -H[0]-H[1]-H[2]+H[3]+J[0]+J[1]-J[2]+J[3]-J[4]-J[5] - logZ )
    Pout[15] = exp( -H[0]-H[1]-H[2]-H[3]+J[0]+J[1]+J[2]+J[3]+J[4]+J[5] - logZ )

    Pout = Pout[::-1]
    return(Pout)
