# ========================================================================================================= #
# Module for solving small n Ising models exactly.
# Author : Edward Lee, edlee@alumni.princeton.edu
#
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
# ========================================================================================================= #
import numpy as np
import scipy.special as ss
from itertools import combinations


def write_eqns(n, sym, corrTermsIx, suffix=''):
    """
    Create strings for writing out the equations and then write them to file.

    Parameters
    ----------
    n : int
        number of spins
    sym : int
        value of 1 will use {-1,1} formulation, 0 means {0,1}
    corrTermsIx : list of ndarrays
        Allows specification of arbitrary correlations to constrain using an index based
        structure. These should be index arrays as would be returned by np.where that
        specify which correlations to write down. Each consecutive array should specify
        a matrix of sequentially increasing dimension.
        [Nx1, NxN, NxNxN, ...]
    suffix : str, ''
    """

    import re
    assert sym in [0,1], "sym must be 0 or 1."
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'
    expterms = [] # 2**N exponential corrTermsIx
    binstates = [] # all binary states as strings
    signs = []  # coefficient for all numerator terms when computing correlations
    br = "[]"
    ix0 = 0
    
    # Collect all corrTermsIx in the partition function.
    for state in range(2**n):
        binstates.append("{0:b}".format(state))
        if len(binstates[state])<n:
            binstates[state] = "0"*(n-len(binstates[state])) + binstates[state]
        expterms.append( '' )

        # Get corrTermsIx corresponding to each of the ith order term.
        if sym:
            for i in range(len(corrTermsIx)):
                expterms[state] += get_terms11(corrTermsIx[i], abc[i], binstates[state], br, ix0)
        else:
            for i in range(len(corrTermsIx)):
                expterms[state] += get_terms01(corrTermsIx[i], abc[i], binstates[state], br, ix0)

        expterms[state] = re.sub(r'\+0\+','+',expterms[state])
        expterms[state] = re.sub(r'\)\+0',')',expterms[state])
        expterms[state] += ', '

    # Collect all terms with corresponding prefix in the equation to solve.
    for state in range(2**n):
        for i in range(len(corrTermsIx)):
            if state==0:
                signs.append([])

            # Get corrTermsIx corresponding to each of the ith order term.
            if sym:
                signs_ = _compute_signs(corrTermsIx[i], expterms[state], binstates[state])
            else:
                signs_ = _compute_signs(corrTermsIx[i], expterms[state], binstates[state], False)
            # expand the length of signs if we haven't reached those constraints yet before
            if len(signs[i])<signs_.size:
                for j in range(signs_.size-len(signs[i])):
                    signs[i].append(np.zeros(0, dtype=int))
            for j in range(signs_.size):
                signs[i][j] = np.append(signs[i][j], signs_[j])

    Z = ''.join(expterms)

    # Account for fact that symmetric Python had inverted the order of the states.
    if sym:
        extra = '\n    Pout = Pout[::-1]'
    else:
        extra = ''
    write_py(n, corrTermsIx, signs, expterms, Z, extra=extra, suffix=suffix)

def write_py(n, contraintTermsIx, signs, expterms, Z, extra='', suffix=''):
    """
    Write out Ising equations for Python.

    Parameters
    ----------
    n : int
        System size.
    contraintTermsIx : list of str
    signs : list of ndarray
        Sign for each term in the numerator when computing correlations.
    expterms : list of str
        Every single energy term.
    Z : str
        Energies for all states that will be put into partition function.
    extra : str, ''
    suffix : str, ''
    extra (str,'') : any extra lines to add at the end
    """

    import time
    import os
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'

    fname = 'ising_eqn/ising_eqn_%d%s.py'%(n,suffix)
    print("Generating file ./%s"%fname)
    if not os.path.isdir('./ising_eqn'):
        os.makedirs('./ising_eqn')
    f = open(fname,'w')
    # insert license
    try:
        license = open('../LICENSE.txt','r').readlines()
        for el in license:
            el = '# '+el
            f.write(el)
        f.write('\n')
    except FileNotFoundError:
        print("License file not found...")

    f.write("# Equations for %d-spin Ising model.\n\n"%n)
    f.write("# ")
    f.write(time.strftime("Written on %Y/%m/%d.")+"\n")
    f.write("from numpy import zeros, exp, array, prod, isnan\nfrom ..enumerate import fast_logsumexp\n\n")

    # Keep these as string because they need to grow in the loop and then can just be
    # added all at once at the end.
    fargs = "def calc_observables(params):\n"
    vardec = '    Cout = zeros(('+str(sum([len(i) for i in signs]))+'))\n' # string of variable declarations
    eqns = '' # string of equations to compute
    ix = np.hstack(( 0, np.cumsum([len(i) for i in signs]) ))

    for i in range(len(contraintTermsIx)):
        vardec += '    '+abc[i]+' = params['+str(ix[i])+':'+str(ix[i+1])+']\n'

    if sym:
        k = 0
        for i in range(len(contraintTermsIx)):
            for j in range(len(signs[i])):
                eqns += ("    num = fast_logsumexp(energyTerms, "+
                         str(signs[i][j]).replace('1 ','1,').replace('1\n','1,\n')+
                         ")\n    Cout["+str(k)+"] = exp( num[0] - logZ ) * num[1]\n")
                k += 1
    else:
        k = 0
        for i in range(len(contraintTermsIx)):
            for j in range(len(signs[i])):
                eqns += ("    num = fast_logsumexp(energyTerms, "+
                 str(signs[i][j]).replace('0 ','0,').replace('1 ','1,').replace('0\n','0,\n').replace('1\n','1,\n')+
                 ")\n    Cout["+str(k)+"] = exp( num[0] - logZ ) * num[1]\n")
                k += 1
    
    # Write out correlation terms
    f.write(fargs)
    f.write(("    \"\"\"\n    Give all parameters concatenated into one array from lowest to highest order.\n"+
             "    Returns all correlations.\n    \"\"\"\n"))
    f.write(vardec)
    _write_energy_terms(f, Z)
    f.write(eqns)
    f.write("    Cout[isnan(Cout)] = 0.\n")
    f.write("    return(Cout)\n\n")

    # Write equations for probabilities of all states.
    #f.write("def p("+string.join([i+"," for i in abc[:len(contraintTermsIx)]])+"):\n")
    f.write("def p(params):\n")
    f.write(("    \"\"\"\n    Give all parameters concatenated into one array from lowest to highest order.\n"+
             "    Returns probabilities of all configurations.\n    \"\"\"\n"))
    f.write(vardec)
   
    # Output variable decs and put params into explicit parameters.
    ix = np.hstack(( 0, np.cumsum([len(i) for i in signs]) ))
    vardec = ''
    for i in range(len(contraintTermsIx)):
        vardec += '    '+abc[i]+' = params['+str(ix[i])+':'+str(ix[i+1])+']\n'
    vardec += '    Pout = zeros(('+str(2**n)+'))\n' # string of variable declarations
    f.write(vardec)
    _write_energy_terms(f, Z)
    
    # each probability equation
    for i in range(len(expterms)):
        f.write('    Pout['+str(i)+'] = exp( '+expterms[i][:-2]+' - logZ )\n')

    f.write(extra)
    f.write("\n    return(Pout)\n")
    f.close()

def _write_energy_terms(f, Z):
    """Split expression for energy terms for each term in Z into multiple lines and write
    out nicely into file.
    
    Parameters
    ----------
    f : file
    Z : list of str
        Energy terms to write out.
    """

    f.write('    energyTerms = array([')
    i=0
    while i<len(Z):
        iend=i+100
        # end line on a +
        while iend<len(Z) and Z[iend-1]!='+':
            iend+=1
        if iend>=len(Z):
            # ignore comma at end of line
            f.write('            '+Z[i:-1]+'])\n    logZ = fast_logsumexp(energyTerms)[0]\n')
        else:
            f.write('    '+Z[i:iend]+'\n')
        i=iend

def _compute_signs(subix, expterm, binstate, sym=True):
    """Iterate through terms that belong in the numerator for each constraint and keep
    track of the sign of those terms.
    
    Parameters
    ----------
    subix : list
    expterm : list of str
    binstate : list of str
    sym : bool, True

    Returns
    -------
    ndarray
        Sign of each exponential term in numerator.
    """

    if len(subix)==0:
        return
    
    if sym:
        downSpin = -1
        signs = np.ones(len(subix[0]), dtype=int)

        for i in range(len(subix[0])):
            if np.mod( sum([binstate[k[i]]=="1" for k in subix]),2 ):
                signs[i] = downSpin
    else:
        downSpin = 0
        signs = np.ones(len(subix[0]), dtype=int)

        for i in range(len(subix[0])):
            if np.mod( any([binstate[k[i]]=="0" for k in subix]),2 ):
                signs[i] = downSpin
    return signs

def get_terms11(subix, prefix, binstate, br, ix0):
    """
    Specific to {-1,1}.
    """
    j = 0
    s = ''
    if len(subix)==0:
        return s
    for i in range(len(subix[0])):
        if np.mod( sum([binstate[k[j]]=="1" for k in subix]),2 ):
            s += '-'
        else:
            s += '+'
        s += prefix+br[0]+str(j+ix0)+br[1]
        j += 1

    return s

def get_terms01(subix, prefix, binstate, br, ix0):
    """
    Specific to {0,1}.
    """
    j = 0
    s = ''
    if len(subix)==0:
        return s
    for i in range(len(subix[0])):
        if np.all( [binstate[k[j]]=="1" for k in subix] ):
            s += '+'+prefix+br[0]+str(j+ix0)+br[1]
        j += 1

    if s=='':
        s = '+0'

    return s

def get_terms(subix, prefix, binstate, br, ix0):
    """
    Spins are put in explicitly
    """
    j = 0
    s = ''
    if len(subix)==0:
        return s
    for i in range(len(subix[0])):
        s += '+'+prefix+br[0]+str(j+ix0)+br[1]
        for k in range(len(subix)):
                s += '*s'+br[0]+str(subix[k][i])+br[1]
        j += 1

    if s=='':
        s = '+0'

    return s

def get_3idx(n):
    """Get binary 3D matrix with truth values where index values correspond to the index
    of all possible ijk parameters.  We can do this by recognizing that the pattern along
    each plane in the third dimension is like the upper triangle pattern that just moves
    up and over by one block each cut lower into the box.
    """

    b = np.zeros((n,n,n))
    c = np.triu(np.ones((n-1,n-1))==1,1)
    for i in range(n-1):
        # shunt this diagonal matrix over every descent into a lower plane in the box
        # the plane xz
        if i==0:
            b[i,(1+i):,(1+i):] = c
        else:
            b[i,(1+i):,(1+i):] = c[:-i,:-i]
    return b

def get_nidx(k, n):
    """
    Get the kth order indices corresponding to all the states in which k elements
    are firing up out of n spins. The ordering correspond to that returned by
    bin_states().

    One can check this code for correctness by comparing with get_3idx()
    >>>>>
    print where(exact.get_3idx(4))
    print where(exact.get_nidx(3,4))
    <<<<<
    """
    
    if k==n:
        return np.reshape(list(range(n)),(n,1))
    elif k<n:
        allStates = bin_states(n)
        statesix = np.sum(allStates,1)==k
        ix = []
        for s in allStates[statesix,:]:
            j = 0
            for i in np.argwhere(s==1).flatten():
                if len(ix)<(j+1):
                    ix.append([])
                ix[j].append(i)
                j += 1
        return np.array(ix)[:,::-1] # make sure last idx increases first

def pairwise(n, sym=0):
    assert sym==0 or sym==1

    print("Writing equations for pairwise Ising model with %d spins."%n)
    if sym:
        write_eqns(n, sym, [np.where(np.ones((n))==1),
                            np.where(np.triu(np.ones((n,n)),k=1)==1)],
                   suffix='_sym')
    else:
        write_eqns(n, sym, [np.where(np.ones((n))==1),
                            np.where(np.triu(np.ones((n,n)),k=1)==1)])

def triplet(n, sym=0):
    assert sym==0 or sym==1

    print("Writing equations for Ising model with triplet interactions and %d spins."%n)
    if sym:
        write_eqns(n,sym,[(range(n),),
                          list(zip(*list(combinations(range(n),2)))),
                          list(zip(*list(combinations(range(n),3))))], suffix='_sym_triplet')
    else:
        write_eqns(n,sym,[(range(n),),
                          list(zip(*list(combinations(range(n),2)))),
                          list(zip(*list(combinations(range(n),3))))], suffix='_triplet')

def _write_matlab(n, terms, fitterms, expterms, Z, suffix=''):
    """
    DEPRECATED: code here for future referencing
    Write out equations to solve for matlab.
    """

    import time
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'
    vardec = ''

    # Write function to solve to file.
    f = open('ising_eqn_%d%s.m'%(n,suffix),'w')
    f.write("% Equations of %d-spin Ising model.\n\n"%n)
    f.write(time.strftime("%Y/%m/%d")+"\n")
    f.write("% Give each set of parameters concatenated into one array.\n\n")

    # Keep these as string because they need to grow in the loop and then can just be
    # added all at once at the end.
    f.write("function Cout = calc_observables(params)\n")
    f.write('\tCout = zeros('+str(sum([len(i) for i in fitterms]))+',1);\n') # string of variable declarations
    eqns = '' # string of equations to compute
    ix = np.hstack(( 0,np.cumsum([len(i) for i in fitterms]) ))+1

    for i in range(len(terms)):
        vardec += '\t'+abc[i]+' = params('+str(ix[i])+':'+str(ix[i+1]-1)+');\n'
    k = 0
    for i in range(len(terms)):
        for j in range(len(fitterms[i])):
            eqns += "\tCout("+str(k+1)+") = ("+fitterms[i][j]+")/Z;\n"
            k += 1

    f.write(vardec)
    f.write("\tZ = "+Z+";\n")
    f.write(eqns)
    f.close()

    g = open('probs'+str(n)+'.m','w')
    g.write("% File for getting the probabilities of Ising model.\n% ")
    g.write(time.strftime("%Y/%m/%d")+"\n")
    # Write equations for probabilities of all states.
    g.write("function Pout = p(params)\n")
    g.write(vardec)
    g.write('    Pout = zeros('+str(2**n)+',1);\n') # string of variable declarations

    g.write('    Z = '+Z+';\n')
    for i in range(len(expterms)):
        g.write('    Pout('+str(i+1)+') = '+expterms[i]+'/Z;\n')

    g.close()

def fast_logsumexp(X, coeffs=None):
    """Simplified version of logsumexp to do correlation calculation in Ising equation
    files. Scipy's logsumexp can be around 10x slower in comparison.
    
    Parameters
    ----------
    X : ndarray
        Terms inside logs.
    coeffs : ndarray
        Factors in front of exponentials. 

    Returns
    -------
    float
        Value of magnitude of quantity inside log (the sum of exponentials).
    float
        Sign.
    """

    Xmx = max(X)
    if coeffs is None:
        y = np.exp(X-Xmx).sum()
    else:
        y = np.exp(X-Xmx).dot(coeffs)

    if y<0:
        return np.log(np.abs(y))+Xmx, -1.
    return np.log(y)+Xmx, 1.


if __name__=='__main__':
    """
    When run with Python, this will write the equations for the Ising model
    into file ising_eqn_[n][_sym] where n will be replaced by the system size
    and the suffix '_sym' is included if the equations are written in the
    {-1,+1} basis.

    To write the Ising model equations for a system of size 3 in the {0,1} basis, call
    >>> python enumerate.py 3

    For the {-1,1} basis, call
    >>> python enumerate.py 3 1

    To include triplet order interactions, include a 3 at the very end
    >>> python enumerate.py 3 0 3
    """

    import sys
    n = int(sys.argv[1])
    if len(sys.argv)==2:
        sym = 0
        order = 2
    elif len(sys.argv)==3:
        sym = int(sys.argv[2])
        assert sym==0 or sym==1
        order = 2
    elif len(sys.argv)==4:
        sym = int(sys.argv[2])
        order = int(sys.argv[3])
    else:
        raise Exception("Unrecognized arguments.")
    
    if order==2:
        pairwise(n, sym) 
    elif order==3:
        triplet(n, sym) 
    else:
        raise NotImplementedError("Only order up to 3 implemented for this convenient interface.")

