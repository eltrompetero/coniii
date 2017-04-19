# Module for solving small n Ising models exactly.
# Code written by Eddie Lee (edlee@alumni.princeton.edu), provided on webpage
# http://pages.discovery.wisc.edu/~elee/research.html with no guarantees
# whatsoever.
# 2014-08-26

import numpy as np
import scipy.special as ss
import string

def solve_ising(n,C,params0,sym,method='fast', maxParameterValue=50, nTries=5, **kwargs):
    """
    Solve Ising model given pairwise correlations.
    2015-08-19

    Params:
    --------
    n (int)
        suffix of tosolve module
    C (ndarray)
        vector of concatenate((si,sisj))
    params0 (ndarray)
        starting point
    sym (bool)
        If True, will use {+1,-1} formulation.
    method (str, 'fast')
        'slow' is for LMNPACK and 'fast' is for less accurate Powell method
    maxParameterValue (50, float)
        Maximum allowed parameter values so that we don't run into overflow errors while searching for viable
        parameters. May need to cahnge this limit for badly constrained problems.
    """
    import scipy.optimize as sopt
    import importlib
   
    assert C.size==(n+n*(n-1)/2)
    assert C.size==params0.size, "Given constraints must number same as parameters."
    assert n>0
    
    if sym:
        tosolve = importlib.import_module('tosolve11.tosolve%d'%n)
    else:
        tosolve = importlib.import_module('tosolve01.tosolve%d'%n)

    def f(params):
        if (np.abs(params)>maxParameterValue).any():
            return [1e30]*len(params)
        return tosolve.get_stats(params) - C
    
    if method=='slow':
        soln = sopt.leastsq( lambda x: f(x), params0, **kwargs )
        print "Distance is %f\n" %np.linalg.norm( f(soln[0]) )
        return soln
    elif method=='fast':
        i = 0
        soln = sopt.fsolve(lambda x: f(x), params0, **kwargs)
        while i<(nTries-1) and soln[2]!=1:
            soln = sopt.fsolve(lambda x: f(x), params0+np.random.normal(size=len(params0),scale=.1), **kwargs)
            i += 1
        print "Distance is %f\n" %np.linalg.norm( f(soln) )
        return soln
        #return sopt.fmin_slsqp(lambda x: np.abs(f(x,C,tosolve)),np.hstack(params0))
        #return sopt.fmin_tnc(lambda x: np.abs(f(x,C,tosolve)),np.hstack(params0))
        #return sopt.fmin_l_bfgs_b(lambda x: np.abs(f(x,C,tosolve)),np.hstack(params1))
    else:
        raise Exception("Choose valid method.")

def write_eqns(n,sym,terms,writeto="matlab"):
    """
        Args:
            n : number of spins
            sym : value of 1 will use {-1,1} formulation, 0 means {0,1}
            terms : list of numpy index arrays as would be returned by np.where that 
                specify which terms to include, each consecutive array should 
                specify indices in an array with an extra dimension of N, 
                [Nx1,NxN,NxNxN,...]
                note that the last dimension is the first to be iterated
            writeto (str,'matlab') : filetype to choose, 'matlab' or 'python'
        Val:
            None
    2013-12-18
    """
    import re
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'
    expterms = [] # 2**N exponential terms
    fitterms = [] # exponential terms with product of spins prefix for calculating
                  # correlations
    binstates = [] # all binary states as strings
    if writeto=="matlab" or writeto=='m':
        br = "()" # brackets for referring to elements of arrays
        ix0 = 1 # starting index for arrays
    elif writeto=='python' or writeto=='py':
        br = "[]"
        ix0 = 0
    else:
        raise Exception("Invalid option for output file type.")
    
    # Collect all terms in the partition function.
    for state in range(2**n):
        binstates.append("{0:b}".format(state))
        if len(binstates[state])<n:
            binstates[state] = "0"*(n-len(binstates[state])) + binstates[state]
        expterms.append( '+exp(' )
        for i in range(len(terms)):
            # Get terms corresponding to each of the ith order term.
            if sym==1:
                expterms[state] += get_terms11(terms[i],abc[i],binstates[state],br,ix0)
            elif sym==0:
                expterms[state] += get_terms01(terms[i],abc[i],binstates[state],br,ix0)
            else:
                expterms[state] += get_terms(terms[i],abc[i],binstates[state],br,ix0)
            expterms[state] = re.sub('\+0\+','+',expterms[state])
            expterms[state] = re.sub('\)\+0',')',expterms[state])
        expterms[state] += ')'

    # Collect all terms with corresponding prefix in the equation to solve.
    for state in range(2**n):
        for i in range(len(terms)):
            if state==0:
                fitterms.append([])
            # Get terms corresponding to each of the ith order term.
            if sym==1:
                add_to_fitterm11(fitterms[i],terms[i],expterms[state],binstates[state])
            elif sym==0:
                add_to_fitterm01(fitterms[i],terms[i],expterms[state],binstates[state])
            else:
                pass
    Z = string.join(expterms,sep="")

    if writeto=="matlab":
        write_matlab(n,terms,fitterms,expterms,Z)
    elif writeto=="py":
        # Account for fact that symmetric Python had inverted the order of the states.
        if sym==1:
            extra = '\n\tPout = Pout[::-1]'
        else:
            extra = ''
        write_py(n,terms,fitterms,expterms,Z,extra=extra)
    else:
        raise Exception("Must choose between \"matlab\" and \"py\".")

    # print(Z)
#     for i in fitterms:
#         print(i)
    return

def write_matlab(n,terms,fitterms,expterms,Z):
    """
    2013-12-18
        Write out equations to solve for matlab.
    """
    import time
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'
    vardec = ''

    # Write function to solve to file.
    f = open('tosolve'+str(n)+'.m','w')
    f.write("% File for solving the Ising model.\n% ")
    f.write(time.strftime("%Y/%m/%d")+"\n")
    f.write("% Give each set of parameters separately in an array.\n\n")

    # Keep these as string because they need to grow in the loop and then can just be
    # added all at once at the end.
    f.write("function Cout = get_stats(params)\n")
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
    g.write("function Pout = get_probs(params)\n")
    g.write(vardec)
    g.write('\tPout = zeros('+str(2**n)+',1);\n') # string of variable declarations

    g.write('\tZ = '+Z+';\n')
    for i in range(len(expterms)):
        g.write('\tPout('+str(i+1)+') = '+expterms[i]+'/Z;\n')

    g.close()
    return

def write_py(n,terms,fitterms,expterms,Z,extra=''):
    """
    2013-12-18
        Write out equations to solve for Python.
        Args:
            extra (str,'') : any extra lines to add at the end
    """
    import time
    abc = 'HJKLMNOPQRSTUVWXYZABCDE'

    # Write function to solve to file.
    f = open('tosolve'+str(n)+'.py','w')
    f.write("# File for solving the Ising model.\n\n")
    f.write("# ")
    f.write(time.strftime("%d/%m/%Y")+"\n")
    f.write("from numpy import zeros, exp\n\n")

    # Keep these as string because they need to grow in the loop and then can just be
    # added all at once at the end.
    fargs = "def get_stats(params):\n"
    vardec = '\tCout = zeros(('+str(sum([len(i) for i in fitterms]))+'))\n' # string of variable declarations
    eqns = '' # string of equations to compute
    ix = np.hstack(( 0,np.cumsum([len(i) for i in fitterms]) ))

    for i in range(len(terms)):
        vardec += '\t'+abc[i]+' = params['+str(ix[i])+':'+str(ix[i+1])+']\n'
    k = 0
    for i in range(len(terms)):
        for j in range(len(fitterms[i])):
            eqns += "\tCout["+str(k)+"] = ("+fitterms[i][j]+")/Z\n"
            k += 1

    f.write(fargs)
    f.write("\t\"\"\"\n\t\tGive each set of parameters separately in an array.\n\t\"\"\"\n")
    f.write(vardec)
    f.write("\tZ = "+Z+"\n")
    f.write(eqns)
    f.write("\n\treturn(Cout)\n\n")

    # Write equations for probabilities of all states.
    #f.write("def get_probs("+string.join([i+"," for i in abc[:len(terms)]])+"):\n")
    f.write("def get_probs(params):\n")
    f.write("\t\"\"\"\n\t\tGive each set of parameters separately in an array.\n\t\"\"\"\n")
   
    # Output variable decs and put params into explicit parameters.
    ix = np.hstack(( 0,np.cumsum([len(i) for i in fitterms]) ))
    vardec = ''
    for i in range(len(terms)):
        vardec += '\t'+abc[i]+' = params['+str(ix[i])+':'+str(ix[i+1])+']\n'
    vardec += '\tPout = zeros(('+str(2**n)+'))\n' # string of variable declarations
    f.write(vardec)

    f.write('\tZ = '+Z+'\n')
    for i in range(len(expterms)):
        f.write('\tPout['+str(i)+'] = '+expterms[i]+'/Z\n')

    f.write(extra)
    f.write("\n\treturn(Pout)\n")
    f.close()
    return

def add_to_fitterm11(fitterm,subix,expterm,binstate):
    """
    2013-12-05
    """
    if len(subix)==0:
        return
    j = 0
    for i in range(len(subix[0])):
        if len(fitterm)>j:
            if np.mod( sum([binstate[k[j]]=="1" for k in subix]),2 ):
                fitterm[j] += expterm+'*-1'
            else:
                fitterm[j] += expterm
        else:
            if np.mod( sum([binstate[k[j]]=="1" for k in subix]),2 ):
                fitterm.append(expterm+'*-1')
            else:
                fitterm.append(expterm)
        j+=1
    return

def add_to_fitterm01(fitterm,subix,expterm,binstate):
    """
    2013-12-05
    """
    if len(subix)==0:
        return
    for i in range(len(subix[0])):
        if len(fitterm)<len(subix[0]):
            fitterm.append('')
        # If all members of the relevant tuple are ==1, include term.
        if np.all( [binstate[k[i]]=="1" for k in subix] ):
            fitterm[i] += expterm
    return

def get_terms11(subix,prefix,binstate,br,ix0):
    """
    2013-12-04
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

    return(s)

def get_terms01(subix,prefix,binstate,br,ix0):
    """
    2013-12-04
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

    return(s)

def get_terms(subix,prefix,binstate,br,ix0):
    """
    2013-12-04
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

    return(s)

def get_3idx(n):
    """
    2013-12-18
        Get binary 3D matrix with truth values where index values correspond to the index
        of all possible ijk parameters.
        We can do this by recognizing that the pattern along each plane in the third
        dimension is like the upper triangle pattern that just moves up and over by one
        block each cut lower into the box.
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

def get_nidx(k,n):
    """
    2014-08-22
        Get the kth order indices corresponding to all the states in which k elements
        are firing up out of n spins. The ordering correspond to that returned by
        entropy.get_all_states().

        One can check this code for correctness by comparing with get_3idx()
        Example:
            print where(exact.get_3idx(4))
            print where(exact.get_nidx(3,4))
    """
    if k==n:
        return np.reshape(range(n),(n,1))
    elif k<n:
        from entropy.entropy import get_all_states
        allStates = get_all_states(n)
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
        
