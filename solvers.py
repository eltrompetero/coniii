# Module for class-based solvers for different Inverse Ising methods.
from __future__ import division
import entropy.entropy as entropy
import scipy.optimize as opt
from multiprocessing import Pool,Array,Queue,Process
from misc.utils import unique_rows
from numba import jit
from utils import *



class Solver(object):
    """
    Base class for declaring common methods and attributes.

    Params:
    -------
    n (int)
        System size.
    constraints (ndarray)
    calc_e (function)
        lambda samples,params: return energy
    calc_observables (function)
        For MCH: lambda samples: return observables
        For exact: lambda params: return observables
    sample_method (str)
        Type of sample method. Current options are 'wolff', 'metropolis', 'remc'.
    mch_approximation (function=None)
    multipliers (ndarray=None)
    nJobs (int=None)

    Attributes:
    -----------
    constraints (ndarray)
    calc_e (function)
        with args (sample,parameters) where sample is 2d
    calc_observables (function)
        takes in samples as argument
    mch_approximation (function)
    sampleSize (int)
    multipliers (ndarray)
        set the Langrangian multipliers
    """
    def __init__(self, n, constraints, calc_e, calc_observables,sample_method,
                 multipliers=None,
                 nJobs=None):
        # Do basic checks on the inputs.
        assert type(n) is int
        
        self.n = n
        self.constraints = constraints
        if multipliers is None:
            self.multipliers = np.zeros(constraints.shape)
        else:
            self.multipliers = multipliers
        self.calc_e = calc_e
        self.calc_observables = calc_observables
        
        self.nJobs = nJobs

    def solve(self):
        return
              
    def estimate_jac(self,eps=1e-3):
        """
        Jacobian is an n x n matrix where each row corresponds to the behavior
        of fvec wrt to a single parameter.
        For calculation, seeing Voting I pg 83
        """
        dlamda = np.zeros(self.multipliers.shape)
        jac = np.zeros((self.multipliers.size,self.multipliers.size))
        print "evaluating jac"
        for i in xrange(len(self.multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac
# end Solver


class Exact(Solver):
    """
    Class for solving +/-1 symmetric Ising model maxent problems by gradient descent with flexibility to put
    in arbitrary constraints.  I chose the symmetric model since that seems more natural for parameter
    interpretation and it does not usually take up much more space for code.

    Params:
    -------
    n (int)
        System size.
    constraints (ndarray)
    calc_e (function)
        lambda samples,params: return energy
    calc_observables (function)
        For exact: lambda params: return observables

    Attributes:
    -----------
    constraints (ndarray)
    calc_e (function)
        with args (sample,parameters) where sample is 2d
    calc_observables (function)
        takes in samples as argument
    multipliers (ndarray)
        set the Langrangian multipliers
    """
    def __init__(self, *args, **kwargs):
        super(Solver,self).__init__(*args,**kwargs)

    def solve(self,
              lamda0=None,
              tol=None,
              tolNorm=None,
              disp=False,
              max_param_value=50,
              fsolve_kwargs={}):
        """
        Params:
        ------
        lamda0 (ndarray=None)
            initial starting point
        tol (float=None)
            maximum error allowed in any observable
        tolNorm (float)
            norm error allowed in found solution
        nIters (int=30)
            number of iterations to make when sampling
        disp (bool=False)
        fsolve_kwargs (dict={})

        Returns:
        --------
        """
        if not lamda0 is None:
            assert len(lamda0)==len(self.multipliers)
            self.multipliers = lamda0.copy()
        else: lamda0 = np.zeros((len(self.multipliers)))
        
        def f(params):
            if np.any(np.abs(params)>max_param_value):
                return [1e30]*len(params)
            return self.calc_observables(params)-self.constraints

        return opt.leastsq(f,lamda0,**fsolve_kwargs)

    def estimate_jac(self,eps=1e-3):
        """
        Jacobian is an n x n matrix where each row corresponds to the behavior
        of fvec wrt to a single parameter.
        For calculation, seeing Voting I pg 83
        2015-08-14
        """
        dlamda = np.zeros(self.multipliers.shape)
        jac = np.zeros((self.multipliers.size,self.multipliers.size))
        print "evaluating jac"
        for i in xrange(len(self.multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac

    def setup_sampler(self,
                      sample_method=None,
                      sampler_kwargs={},
                      optimize_kwargs={}):
        """
        Setup sampler.
        2017-04-03
        """
        sample_method = sample_method or self.sampleMethod
        
        if sample_method=='wolff':
            raise NotImplementedError("Need to update call.")
            h,J = self.multipliers[:self.n],self.multipliers[self.n:]
            self.sampler = WolffIsing( J,h )

        elif sample_method=='metropolis':
            raise NotImplementedError("Need to update call.")
            self.sampler = MCIsing( self.n,self.multipliers,self.calc_e )
            
        elif sample_method=='remc':
            self.sampler = ParallelTempering( self.n,
                                              self.multipliers,
                                              self.calc_e,
                                              sampler_kwargs['temps'],
                                              sample_size=self.sampleSize )
            # Parallel tempering needs to optimize choice of temperatures.
            self.sampler.optimize(**optimize_kwargs)
            
        else:
           raise NotImplementedError("Unrecognized sampler.")

    def generate_samples(self,n_iters,
                         sampleSize=None,
                         sample_method=None,
                         initial_sample=None,
                         generate_kwargs={}):
        """
        Wrapper around generate_samples_parallel() from available samplers.
        2017-04-03
        """
        assert not (self.sampler is None), "Must call setup_sampler() first."

        sample_method = sample_method or self.sampleMethod
        sampleSize = sampleSize or self.sampleSize
        if initial_sample is None:
            initial_sample = self.samples
        
        if sample_method=='wolff':
            self.sampler.update_parameters(self.multipliers[self.n:],self.multipliers[:self.n])
            self.samples = self.sampler.generate_sample_parallel( sampleSize,n_iters,
                                                                  initial_sample=initial_sample )

        elif sample_method=='metropolis':
            self.sampler.theta = self.multipliers
            self.sampler.generate_samples_parallel( sampleSize,
                                                    n_iters=n_iters,
                                                    cpucount=cpucount,
                                                    initial_sample=initial_sample )
            self.samples = self.sampler.samples

        elif sample_method=='remc':
            self.sampler.update_parameters(self.multipliers)
            self.sampler.generate_samples(n_iters=n_iters,**generate_kwargs)
            self.samples = self.sampler.replicas[0].samples

        else:
           raise NotImplementedError("Unrecognized sampler.")

    def learn_parameters_mch(self, thisConstraints,
                             maxdlamda=1,
                             maxdlamdaNorm=1, 
                             maxLearningSteps=50,
                             eta=1 ):
        """
        2015-08-14

        Params:
        -------
        thisConstraints (ndarray)
        maxdlamda (1,float)
        maxdlamdaNorm (1,float),
        maxLearningSteps (int)
            max learning steps before ending MCH
        eta (1,float)
            factor for changing dlamda
        """
        keepLearning = True
        dlamda = np.zeros((self.constraints.size))
        learningSteps = 0
        distance = 1
        
        while keepLearning:
            # Get change in parameters.
            # If observable is too large, then corresponding energy term has to go down 
            # (think of double negative).
            dlamda += -(thisConstraints-self.constraints) * np.min([distance,1.]) * eta
            #dMultipliers /= dMultipliers.max()
            
            # Predict distribution with new parameters.
            thisConstraints = self.mch_approximation( self.samples, dlamda )
            distance = np.linalg.norm( thisConstraints-self.constraints )
                        
            # Counter.
            learningSteps += 1

            # Evaluate exit criteria.
            if np.linalg.norm(dlamda)>maxdlamdaNorm or np.any(np.abs(dlamda)>maxdlamda):
                keepLearning = False
            elif learningSteps>maxLearningSteps:
                keepLearning = False

        self.multipliers += dlamda
        return thisConstraints
# End GeneralMaxentSolver

class GeneralMaxentSolver():
    """
    Class for solving +/-1 symmetric Ising model maxent problems by gradient descent with flexibility to put
    in arbitrary constraints.  I chose the symmetric model since that seems more natural for parameter
    interpretation and it does not usually take up much more space for code.

    Params:
    -------
    n (int)
        System size.
    constraints (ndarray)
    calc_e (function)
        lambda samples,params: return energy
    calc_observables (function)
        For MCH: lambda samples: return observables
        For exact: lambda params: return observables
    sample_method (str)
        Type of sample method. Current options are 'wolff', 'metropolis', 'remc'.
    mch_approximation (function=None)
    multipliers (ndarray=None)
    nJobs (int=None)

    Attributes:
    -----------
    constraints (ndarray)
    calc_e (function)
        with args (sample,parameters) where sample is 2d
    calc_observables (function)
        takes in samples as argument
    mch_approximation (function)
    sampleSize (int)
    multipliers (ndarray)
        set the Langrangian multipliers
    """
    def __init__(self, n, constraints, calc_e, calc_observables,sample_method,
                 mch_approximation=None,
                 sampleSize=1000,
                 multipliers=None,
                 nJobs=None):
        assert type(n) is int

        self.n = n
        self.constraints = constraints
        if multipliers is None:
            self.multipliers = np.zeros(constraints.shape)
        else:
            self.multipliers=multipliers
        self.calc_e = calc_e
        self.calc_observables = calc_observables
        self.mch_approximation = mch_approximation
        self.sampleSize = sampleSize
        
        self.sampleMethod = sample_method
        self.sampler = None
        self.samples = None
        self.E = None
        self.nJobs = nJobs

    def solve(self,
              lamda0=None,
              method='slow',
              tol=None,
              tolNorm=None,
              nIters=30,
              maxiter=10,
              disp=False,
              max_param_value=50,
              learn_params_kwargs={},
              generate_kwargs={},
              fsolve_kwargs={}):
        """
        Solve for parameters using MCH routine. Commented part relies on gradient descent but doesn't seem to
        be very good at converging to the right answer with some tests on small systems.
        2017-04-04

        Params:
        ------
        lamda0 (ndarray=None)
            initial starting point
        tol (float=None)
            maximum error allowed in any observable
        tolNorm (float)
            norm error allowed in found solution
        nIters (int=30)
            number of iterations to make when sampling
        disp (bool=False)
        learn_parameters_kwargs
        generate_kwargs
        """
        if not lamda0 is None:
            assert len(lamda0)==len(self.multipliers)
            self.multipliers=lamda0.copy()
        else: lamda0 = np.zeros((len(self.multipliers)))
        tol = tol or 1/np.sqrt(self.sampleSize)
        tolNorm = tolNorm or np.sqrt( 1/self.sampleSize )*len(self.multipliers)
        errors = []
        
        assert tol>0 and tolNorm>0 and nIters>0
        
        if method=='slow':
            self.generate_samples(nIters,
                                  generate_kwargs=generate_kwargs)
            thisConstraints = self.calc_observables(self.samples)
            errors.append( thisConstraints-self.constraints )
            if disp=='detailed': print self.multipliers
            
            # MCH iterations.
            counter=0
            keepLoop=True
            while keepLoop:
                if disp:
                    print "Iterating parameters with MCH..."
                self.learn_parameters_mch(thisConstraints,**learn_params_kwargs)
                if disp=='detailed':
                    print "After MCH step, the multipliers are..."
                    print self.multipliers
                if disp:
                    print "Sampling..."
                self.generate_samples( nIters,
                                       generate_kwargs=generate_kwargs )
                thisConstraints = self.calc_observables(self.samples)
                counter += 1
                
                # Exit criteria.
                errors.append( thisConstraints-self.constraints )
                if ( np.linalg.norm(errors[-1])<tolNorm
                     and np.all(np.abs(thisConstraints-self.constraints)<tol) ):
                    print "Solved."
                    errflag=0
                    keepLoop=False
                elif counter>maxiter:
                    print "Over maxiter"
                    errflag=1
                    keepLoop=False

            return self.multipliers,errflag,np.vstack((errors))

        elif method=='exact':
            # In the case where we enumerate the energies of every state exactly.
            def f(params):
                if np.any(np.abs(params)>max_param_value):
                    return [1e30]*len(params)
                return self.calc_observables(params)-self.constraints

            return opt.leastsq(f,lamda0,**fsolve_kwargs)
        else:
            raise Exception("Invalid solving method.")
        #def f(lamda):
        #    if np.any(np.abs(lamda)>10):
        #        return [1e30]*len(lamda)
        #    self.generate_samples(nIters=20)
        #    print "generating samples for"
        #    print lamda
        #    thisConstraints = self.calc_observables(self.samples)
        #    return thisConstraints-self.constraints

        #if lamda0 is None:
        #    lamda0 = self.multipliers
        #soln = opt.leastsq(f, lamda0, Dfun=lambda x: self.estimate_jac(), full_output=True,**kwargs)
        #self.multipliers = soln[0]
        #return soln

    def estimate_jac(self,eps=1e-3):
        """
        Jacobian is an n x n matrix where each row corresponds to the behavior
        of fvec wrt to a single parameter.
        For calculation, seeing Voting I pg 83
        2015-08-14
        """
        dlamda = np.zeros(self.multipliers.shape)
        jac = np.zeros((self.multipliers.size,self.multipliers.size))
        print "evaluating jac"
        for i in xrange(len(self.multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac

    def setup_sampler(self,
                      sample_method=None,
                      sampler_kwargs={},
                      optimize_kwargs={}):
        """
        Setup sampler.
        2017-04-03
        """
        sample_method = sample_method or self.sampleMethod
        
        if sample_method=='wolff':
            raise NotImplementedError("Need to update call.")
            h,J = self.multipliers[:self.n],self.multipliers[self.n:]
            self.sampler = WolffIsing( J,h )

        elif sample_method=='metropolis':
            raise NotImplementedError("Need to update call.")
            self.sampler = MCIsing( self.n,self.multipliers,self.calc_e )
            
        elif sample_method=='remc':
            self.sampler = ParallelTempering( self.n,
                                              self.multipliers,
                                              self.calc_e,
                                              sampler_kwargs['temps'],
                                              sample_size=self.sampleSize )
            # Parallel tempering needs to optimize choice of temperatures.
            self.sampler.optimize(**optimize_kwargs)
            
        else:
           raise NotImplementedError("Unrecognized sampler.")

    def generate_samples(self,n_iters,
                         sampleSize=None,
                         sample_method=None,
                         initial_sample=None,
                         generate_kwargs={}):
        """
        Wrapper around generate_samples_parallel() from available samplers.
        2017-04-03
        """
        assert not (self.sampler is None), "Must call setup_sampler() first."

        sample_method = sample_method or self.sampleMethod
        sampleSize = sampleSize or self.sampleSize
        if initial_sample is None:
            initial_sample = self.samples
        
        if sample_method=='wolff':
            self.sampler.update_parameters(self.multipliers[self.n:],self.multipliers[:self.n])
            self.samples = self.sampler.generate_sample_parallel( sampleSize,n_iters,
                                                                  initial_sample=initial_sample )

        elif sample_method=='metropolis':
            self.sampler.theta = self.multipliers
            self.sampler.generate_samples_parallel( sampleSize,
                                                    n_iters=n_iters,
                                                    cpucount=cpucount,
                                                    initial_sample=initial_sample )
            self.samples = self.sampler.samples

        elif sample_method=='remc':
            self.sampler.update_parameters(self.multipliers)
            self.sampler.generate_samples(n_iters=n_iters,**generate_kwargs)
            self.samples = self.sampler.replicas[0].samples

        else:
           raise NotImplementedError("Unrecognized sampler.")

    def learn_parameters_mch(self, thisConstraints,
                             maxdlamda=1,
                             maxdlamdaNorm=1, 
                             maxLearningSteps=50,
                             eta=1 ):
        """
        2015-08-14

        Params:
        -------
        thisConstraints (ndarray)
        maxdlamda (1,float)
        maxdlamdaNorm (1,float),
        maxLearningSteps (int)
            max learning steps before ending MCH
        eta (1,float)
            factor for changing dlamda
        """
        keepLearning = True
        dlamda = np.zeros((self.constraints.size))
        learningSteps = 0
        distance = 1
        
        while keepLearning:
            # Get change in parameters.
            # If observable is too large, then corresponding energy term has to go down 
            # (think of double negative).
            dlamda += -(thisConstraints-self.constraints) * np.min([distance,1.]) * eta
            #dMultipliers /= dMultipliers.max()
            
            # Predict distribution with new parameters.
            thisConstraints = self.mch_approximation( self.samples, dlamda )
            distance = np.linalg.norm( thisConstraints-self.constraints )
                        
            # Counter.
            learningSteps += 1

            # Evaluate exit criteria.
            if np.linalg.norm(dlamda)>maxdlamdaNorm or np.any(np.abs(dlamda)>maxdlamda):
                keepLearning = False
            elif learningSteps>maxLearningSteps:
                keepLearning = False

        self.multipliers += dlamda
        return thisConstraints
# End GeneralMaxentSolver


# ============
# MPF Solver #
# ============

def unwrap_self_worker_obj(arg, **kwarg):
    return MPFSolver.worker_objective_task(*arg, **kwarg)

class MPFSolver(object):
    def __init__(self, calc_e, adj, calc_de=None, nWorkers=0 ):
        """
        Slowest step is the computation of the energy of a given state. Make this as fast as possible
        2015-12-26

        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        nWorkers (-1, int)
            if 0 no parallel processing, other numbers above 0 specify number of cores to use
        """
        self.calc_e = calc_e
        self.calc_de = calc_de
        self.adj = adj
        self.nWorkers = nWorkers
        if nWorkers>0:
            self.solve = self.solve_parallel
        
    @staticmethod
    def worker_objective_task( s, Xcount, adjacentStates, params, calc_e ):
        return Xcount * np.sum(np.exp( .5*(calc_e(s[None,:],params) 
                                           - calc_e(adjacentStates,params) ) ))
 
    def K( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute objective function.
        2015-12-26
        
        Params:
        -------
        Xuniq (ndata x ndims ndarray)
            unique states that appear in the data
        Xcount (ndarray of ints)
            number of times that each unique state appears in the data
        adjacentStates (list of ndarrays)
            list of adjacent states for each given unique state
        params (ndarray)
            parameters for computation of energy
        """
        if self.pool is None:
            obj = 0.
            objGrad = np.zeros((params.size))
            for i,s in enumerate(Xuniq):
                dobj = Xcount[i] * np.exp( .5*(self.calc_e(s[None,:],params) 
                                               - self.calc_e(adjacentStates[i],params) ) )
                if not self.calc_de is None:
                    for j in xrange(params.size):
                        if dobj.size!=adjacentStates[i].shape[0]:
                            raise Exception("Sizes do not match")
                        objGrad[j] += .5 * (dobj * ( self.calc_de(s[None,:],j) 
                                            - self.calc_de(adjacentStates[i],j) )).sum()
                obj += dobj.sum()
        else:
            # Parallel loop through objective function calculation for each state in the data.
            obj = [self.pool.apply( unwrap_self_worker_obj, 
                                    args=([Xuniq[i],Xcount[i],adjacentStates[i],params,self.calc_e],) ) 
                        for i in xrange(Xuniq.shape[0])]
            obj = obj.sum()

            if not self.calc_de is None:
                from warning import warn
                warn("Gradient computation not written fro parallel loop.")

        if not self.calc_de is None:
            return obj / Xcount.sum(), objGrad / Xcount.sum()
        else:
            return obj / Xcount.sum()
       
    def _K( self, X, J ):
        """
        Translation from Sohl-Dickstein's code K_dk_ising.m. This is here for testing purposes only.
        Caution: This uses a different convention for negatives and 1/2 factors. To use this properly, all
        parameters will have an extra negative, the returned J's will be halved and the energy calculation
        should include a 1/2 factor in front of h's.
        """
        nbatch, ndims = X.shape
        X = X.T
        
        h = J[:ndims]
        J = squareform( J[ndims:] )
        J[diag_indices(ndims)] = h
        
        Y = dot(J,X)
        diagJ = J.diagonal()
    #     % XnotX contains (X - [bit flipped X])
        XnotX = 2.*X-1;
    #     % Kfull is a [ndims, nbatch] matrix containing the contribution to the 
    #     % objective function from flipping each bit in the rows, for each datapoint 
    #     % on the columns
        Kfull = np.exp( XnotX * Y - (1/2)*tile(diagJ[:,None],(1,nbatch)) )
        K = sum(Kfull)
        K  = K  / nbatch
        return K

    def logK( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute log of objective function.
        2017-01-15
        
        Params:
        -------
        Xuniq (ndata x ndims ndarray)
            unique states that appear in the data
        Xcount (ndarray of ints)
            number of times that each unique state appears in the data
        adjacentStates (list of ndarrays)
            list of adjacent states for each given unique state
        params (ndarray)
            parameters for computation of energy
        """
        from scipy.misc import logsumexp

        obj = 0.
        objGrad = np.zeros((params.size))
        power=np.zeros((len(Xuniq),len(adjacentStates[0])))  # energy differences
        for i,s in enumerate(Xuniq):
            power[i,:] = .5*( self.calc_e(s[None,:],params) - self.calc_e(adjacentStates[i],params) )
            
        obj=logsumexp( power+np.log(Xcount)[:,None] )
        
        if not self.calc_de is None:
            # coefficients that come out from taking derivative of exp
            for i in xrange(params.size):
                gradcoef=np.zeros((len(Xuniq),len(adjacentStates[0])))  
                for j,s in enumerate(Xuniq): 
                    gradcoef[j,:] = .5 * ( self.calc_de(s[None,:],i) 
                                           - self.calc_de(adjacentStates[j],i) )
                power -= power.max()
                objGrad[i]=(gradcoef*np.exp(power)*Xcount[:,None]).sum()/(np.exp(power)*Xcount[:,None]).sum()

        if not self.calc_de is None:
            if objGrad.size==1:
                raise Exception("")
            return obj / Xcount.sum(), objGrad / Xcount.sum()
        else:
            return obj / Xcount.sum()
    # end logK

    def solve( self, X, 
               initialGuess=None,
               method='L-BFGS-B',
               allConnected=True,
               parameterLimits=100,
               solver_kwargs={'maxiter':100,'disp':True},
               uselog=True,
               ):
        """
        Minimize MPF objective function using scipy.optimize.minimize.
        2017-01-14

        Params:
        -------
        X (ndata x ndim ndarray)
            array of states compatible with given energy and adjacent neighbors functions
        adj (lambda)
            returns adjacent states for any given state
        allConnected (True, bool)
            switch for summing over all states that data sets could be connected to or just summing over
            non-data states (second summation in Eq 10 in Sohl-Dickstein 2011)
        iterate (0, int)
            number of times to try new initial conditions if first try doesn't work. Right now, this is a
            pretty coarse test because the fit can be good even without converging.
        parameterLimits (float)
            some limit to constrain the space that the solver has to search. This is the maximum allowed
            magnitude of any single parameter.
        solver_kwargs (dict)
            For solver

        Value:
        ------
        soln (ndarray)
            found solution to problem
        output (dict)
            full output from minimize solver
        """
        assert parameterLimits>0
        
        if not self.calc_de is None:
            includeGrad = True
        else:
            includeGrad = False
        X = X.astype(float)
        if initialGuess is None:
            initialGuess = entropy.calc_sisj( X, concat=True )
   
        # Get list of unique data states and how frequently they appear.
        Xuniq = X[unique_rows(X)]
        ix = unique_rows(X,return_inverse=True)
        Xcount = np.bincount(ix)
        M,N = Xuniq.shape
        
        adjacentStates = []
        for s in Xuniq:
            adjacentStates.append( self.adj(s) )
            # Remove states already in data.
            if not allConnected:
                ix = np.zeros((s.size))==0
                for i,t in enumerate(adjacentStates[-1]):
                    if np.any(np.all(t[None,:]==Xuniq,1)):
                        ix[i] = False
                if np.sum(ix)==X.shape[1]:
                    raise Exception("This data set does not satisfy MPF assumption that each \
                                    state be connected to at least one non-data state (?)")
                adjacentStates[-1] = adjacentStates[-1][ix]

        # Interface to objective function.
        if uselog:
            def f(params):
                return self.logK( Xuniq, Xcount, adjacentStates, params )
        else:
            def f(params):
                return self.K( Xuniq, Xcount, adjacentStates, params )
        
        # If calc_de has been provided then minimize will use gradient information.
        soln = opt.minimize( f, initialGuess,
                             bounds=[(-parameterLimits,parameterLimits)]*len(initialGuess),
                             method=method, jac=includeGrad, options=solver_kwargs )
        return soln['x'], soln
# End MPFSolver



# ==================================
# Max ent transition matrix solver #
# ==================================
class MaxEntTransitionSolver():
    """
    Brute force solver for maxent transition matrix on Ising model. I think
    this solves the strangely complicated formulation with the product of transition matrices.
    2015-07-25
    """
    def __init__(self,n,iStatesAsInt=None,fStatesAsInt=None, initialP=None,finalP=None,distP=None ):
        """
        n (int)
        iStatesAsInt (ndarray)
        fStatesAsInt (ndarray)
        """
        assert n>=0
        
        self.n = n
        self.binstates = entropy.bin_states(n)

        if iStatesAsInt is None:
            self.initialP = initialP
            self.finalP = finalP

            self.distP = distP
        else:
            self.iStatesAsInt = iStatesAsInt
            self.fStatesAsInt = fStatesAsInt
           
            self.initialStates = self.binstates[iStatesAsInt].astype(int)
            self.finalStates = self.binstates[fStatesAsInt].astype(int)
            if initialP is None:
                initialP = np.bincount(iStatesAsInt,minlength=2**n)
                self.initialP = initialP / np.sum(initialP)
                
                finalP = np.bincount(fStatesAsInt,minlength=2**n)
                self.finalP = finalP / np.sum(finalP)
            else:
                assert np.all(initialP>=0) and np.isclose(np.sum(initialP),1)
                assert np.all(finalP>=0) and np.isclose(np.sum(finalP),1)
                
                self.initialP = initialP
                self.finalP = finalP
            
            self.distP = self.calc_dist_P( self.initialStates, self.finalStates )
        
        self.T = np.zeros((2**n,2**n))
        
        self.dMat = self.distance_mat() 
       
        self.l = None  # parameters for jump distance averages
        self.g = None  # parameters for final distribution of probabilities

    def distance_mat(self):
        """
        Distances between all pairs of states.
        """
        dmat = np.zeros((2**self.n,2**self.n))
        for i in xrange( 2**self.n-1 ):
            for j in xrange( i+1,2**self.n ):
                dmat[i,j] = np.sum(np.abs( self.binstates[i].astype(int)-self.binstates[j].astype(int) ))
        return dmat + dmat.T
        
    def calc_dist_P( self, initialStates, finalStates, weights=None ):
        from scipy.special import binom

        # Distance transition frequencies.
        if weights is None:
            distFrequencies = np.bincount( np.sum(np.abs(initialStates-finalStates),1),minlength=self.n+1 )
            distFrequencies = distFrequencies / np.sum(distFrequencies)
        else:
            distFrequencies = np.bincount( np.sum(np.abs(initialStates-finalStates),1),minlength=self.n+1, weights=weights )
        
        # If every posibble transition has been made once, then we should expect
        #distFrequencies = np.array([i + binom(self.n,i) for i in range(self.n+1)])
        #distFrequencies += 1
        
        return distFrequencies

    def dist_P_model(self,lDist,g=None):
        """
        Calculate distribution of distance jumps using model parameters.
        2015-08-08

        Params:
        -------
        lDist (ndarray)
            if only this is given, then assuming that all parameters are given in this array
        g (ndarray)
        """
        if g is None:
            lDist,g = lDist[:(self.n)], lDist[(self.n):]
    
        T = self.transition( lDist,g )
        return [np.sum( (T * self.initialP[:,None])[self.dMat==i] )
                for i in range(0,self.n+1)]

    def transition_data( self ):
        """
        Compute transition matrix from data.
        2015-08-08
        """
        T = np.zeros((2**self.n,2**self.n))
        for i in zip(self.iStatesAsInt,self.fStatesAsInt):
            T[i[0],i[1]] += 1
        T = T / (np.sum(T,1)[:,None]+np.nextafter(0,1))

        return T

    def transition( self, lDist, g=None ):
        """
        Compute transition matrix given energy terms. Using convention for energy values. More negative means more likely.
        2015-08-08

        Params:
        -------
        lDist (ndarray)
            length n; corresponding to constraints on jump size when including normalization
        g (ndarray)
            length 2**n-1; corresponding to constraints on final distribution of states when including normalization
        """
        if g is None:
            lDist,g = lDist[:self.n],lDist[self.n:]

        lDist = np.concatenate([[0],lDist])
        distanceEnergy = np.array([lDist[i.astype(int)] for i in self.dMat]) 
        
        colEnergy = np.concatenate([[0],g])
        
        # Produce transition matrix.
        T = np.exp( -distanceEnergy - colEnergy[None,:] )
        return T / np.sum( T,1 )[:,None]

    def setup_rhs(self):
        """Setup symbolic expression for solving transcental equation for T in singelTransition method.
        2015-08-10"""
        import sympy as sy

        # Define symbols.
        TSy = sy.symbols('t0:%d'%(self.n*2**self.n))
        qSy = sy.symbols('q0:%d'%(self.n+1))
        mSy = sy.Matrix((self.dMat==1)*1.)
        for ix in enumerate(np.argwhere(mSy)):
            mSy[ix[1]] = TSy[ix[0]]
        
        # Get system of equations from derivatives.
        matrixSum = sy.Matrix( np.sum([qSy[i]*mSy**i for i in np.arange(self.n)],0).reshape((2**self.n,2**self.n)) )
        rhsSy = sy.Matrix(2**self.n, 2**self.n, [0]*2**(2*self.n))
        for ix in enumerate(np.argwhere(self.dMat==1)):
            rhsSy[tuple(ix[1])] =  matrixSum[tuple(ix[1])].diff(TSy[ix[0]])
        return rhsSy,TSy,qSy
    
    def define_T_fsolve(self,g,q,rhsSy=None,TSy=None,qSy=None):
        """
        Params:
        -------
        g (ndarray)
            Complete 2**n components.
        q (ndarray)
            complete set of n+1 elements.
        """
        if rhsSy is None or TSy is None or qSy is None:
            rhsSy,TSy,qSy = self.setup_rhs()
        def solve_this_for_T(TGuess):
            if np.any(TGuess<0) or np.any(TGuess>1):
                return [1e30]*len(TGuess)
            # NOTE: This error checking could be made faster by instead only setting the relevant
            # row to be 1e30 instead of the whole matrix.
            # Or maybe not...they all mix when the matrix is put to a power.
            rowSums = np.array([ np.sum(TGuess[(i*(self.n-1)):((i+1)*(self.n-1))]) 
                                 for i in xrange(2**self.n) ])
            if np.any( rowSums > 1):
                return [1e30]*len(TGuess)
             
            # Insert remaining elements using normalization.
            TGuess = np.insert( TGuess, range(0,TGuess.size,self.n-1), 1-rowSums ) 
          
            allSubs = zip( np.concatenate([TSy,qSy]),np.concatenate([TGuess,q]) )

            # Normalize guess for T across rows.
            T = np.zeros((2**self.n,2**self.n))
            T[self.dMat==1] = TGuess
            
            # Evaluate RHS of equation as in pg 8 of Voting I.
            rhs = rhsSy.subs(allSubs)
            rhs = np.exp( g[None,:] * np.array(rhs).astype(float)  )  # Is there a negative missing for convention?
            #Normalize across rows.
            rhs = np.array([i[1] / np.sum(i[1][self.dMat[i[0]]==1]) for i in enumerate(rhs)])
            
            result = T[self.dMat==1]-rhs[self.dMat==1] 
            return np.delete(result,range(0,TGuess.size,self.n)) 
        return solve_this_for_T

    def rebuild_T(self,TGuess):
        """
        When we put T into the solver, we only put in n-1 elements for each row (single flips) for normalization, so this is a convenience function for getting the rest of T back.
        2015-08-10

        Value:
        T (ndarray)
            2**n * n elements
        """
        rowSums = np.array([ np.sum(TGuess[(i*(self.n-1)):((i+1)*(self.n-1))]) 
                             for i in xrange(2**self.n) ])
        
        # Insert remaining elements using normalization.
        return np.insert( TGuess, range(0,TGuess.size,self.n-1), 1-rowSums ) 

    def solve_for_T(self,solve_this_for_T,maxIters=10):
        """
        2015-08-10
        Params:
        -------
        solve_this_for_T (function)
            put into fsolve to solve for T
        maxIters (int):
            max iterations allowed for random initial conditionsfro finding T
        """
        foundTSoln = False
        TSolnIterations = 0
        while (not foundTSoln) and (TSolnIterations<maxIters):
            # Make a guess for the transition matrix.
            initialTGuess = np.random.rand(2**self.n*(self.n-1))/self.n

            TSoln = opt.fsolve( solve_this_for_T, initialTGuess, full_output=True )
            if TSoln[2]==1:
                foundTSoln = True
                #print "T is"
                #print TSoln[0]
            TSolnIterations += 1
            
        if TSolnIterations==maxIters:
            print "No T solution found for transcendental equation."
            return [1e30]*self.finalP.size
        return TSoln

    def solve_this( self, prior='flat' ):
        """
        Return function to solve by minimization.
        2015-08-08
        Params:
        -------
        prior (str)
            'flat' prior will match observables exactly
            'normalOne' will add prior from energy differences of states that are distance of 1 apart
            'singleTransition' 
        """
        if prior=='flat':
            def f(params):
                if np.any(np.abs(params)>100):
                    return 1e30
                lDist,g = params[:(self.n)], params[(self.n):]

                T = self.transition( lDist,g )
                
                # Remember that we don't need to worry about normalization, which removes one equation from both sets of parameters.
                finalPDist = np.sum( T * self.initialP[:,None], 0 )[:-1] - self.finalP[:-1]
                distPDist = [np.sum( (T * self.initialP[:,None])[self.dMat==i] ) - self.distP[i] 
                             for i in range(1,self.n+1)]
                
                return np.concatenate([finalPDist,distPDist])**2
        elif prior=='normalOne':
            sigma = 1
            def f(params):
                if np.any(np.abs(params)>100):
                    return 1e30
                lDist,g = params[:self.n], params[self.n:]

                T = self.transition( lDist,g )
                allTransitionProbabilities = T * self.initialP[:,None]
                
                likelihood = np.sum(np.log( allTransitionProbabilities[self.iStatesAsInt,self.fStatesAsInt] ))
                
                pairsIx = np.argwhere( self.dMat==1 )
                g = np.concatenate([[0],g])
                priorCost = -1/(2*sigma**2) * sum([( -np.log(self.finalP[i[1]]/self.finalP[i[0]])+(g[i[1]]-g[i[0]]) )**2 
                                                     for i in pairsIx])
                return -(likelihood+priorCost)
        elif prior=='singleTransition':
            rhsSy,TSy,qSy = self.setup_rhs()
            
            def f(params):
                # Logic: We wish to find set of gamma and q that return final distribution p_f.
                # For each guess of the parameter values g and q, we must solve the transcendental 
                # equation for the transition matrix T. Since the transcendental equation depends on
                # g and q, we redefine it for every new guess of parameters.
                if np.any(np.abs(params)>50):
                    return [1e30]*len(params)
                if np.any(params[(2**self.n-1):]<0) or np.sum(params[(2**self.n-1):])>1:
                    return [1e30]*len(params)

                # Read in parameters.
                g,q = params[:(2**self.n-1)],params[(2**self.n-1):]
                g = np.concatenate([[0],g])
                q = np.concatenate([[1-np.sum(q)],q])
                
                # First must find the transition matrix as given by equation.
                solve_this_for_T = self.define_T_fsolve(g,q,rhsSy=rhsSy,TSy=TSy,qSy=qSy)
                
                TSoln = self.solve_for_T(solve_this_for_T)
                
                # Put found solution in to matrix form
                T = np.zeros((2**self.n,2**self.n))
                T[self.dMat==1] = self.rebuild_T( TSoln[0] )
               
                completeT = np.sum([q[i]*np.linalg.matrix_power(T,i) for i in range(self.n+1)],0)
                return (np.sum( completeT*self.initialP[:,None], 0 ) - self.finalP)[1:]
        else:
            raise Exception("Prior type not recognized.")

        return f
    def solve( self, method='fast', initialGuess=None, prior='flat', q=None, **kwargs ):
        """
        Params:
        -------
        method (str,'fast'):
            fast 
        initialGuess (ndarray,None)
        **kwargs (for solver)
        """
        if initialGuess is None:
            initialGuess = np.random.normal( size=2**self.n + self.n-1 )
        
        if prior=='flat':
            if method=='fast':
                soln = opt.fsolve( self.solve_this(prior), initialGuess, full_output=1, **kwargs )
            else:
                soln = opt.leastsq( self.solve_this(prior), initialGuess, full_output=1, **kwargs )
            self.l = soln[0][:self.n]
            self.g = soln[0][(self.n):]

        elif prior=='normalOne':
            if method=='fast':
                soln = opt.fmin( self.solve_this(prior), initialGuess, full_output=1, **kwargs )
            else:
                soln = opt.leastsq( self.solve_this(prior), initialGuess, full_output=1, **kwargs )
            self.l = soln[0][:self.n]
            self.g = soln[0][(self.n):]

        elif prior=='singleTransition':
            if method=='fast':
                soln = opt.fsolve( lambda g: self.solve_this(prior)(np.concatenate([g,q[1:]])), 
                                   initialGuess, full_output=1, **kwargs ) 
            else:
                soln = opt.leastsq( lambda g: self.solve_this(prior)(np.concatenate([g,q[1:]])), 
                                    initialGuess, full_output=1, **kwargs ) 

            self.g = np.concatenate([[0],soln[0]])
            #soln = opt.fsolve( self.solve_this(prior), initialGuess, full_output=1, **kwargs ) 

        soln = list(soln)
        soln.append(initialGuess)

        return soln


# ========================================= #
# Helper functions for solving Ising model. # 
# ========================================= #
def define_ising_mch_helpers():
    """
    Functions for plugging into GeneralMaxentSolver for solving +/-1 Ising model.

    Returns:
    --------
    calc_e,mch_approximation
    """
    from mc_hist import GeneralMaxentSolver
    from entropy.entropy import calc_sisj

    # Defime functions necessary for solving.
    @jit(nopython=True,cache=True)
    def fast_sum(J,s):
        e = np.zeros((s.shape[0]))
        for n in xrange(s.shape[0]):
            k = 0
            for i in xrange(s.shape[1]-1):
                for j in xrange(i+1,s.shape[1]):
                    e[n] += J[k]*s[n,i]*s[n,j]
                    k += 1
        return e

    def calc_e(s,params):
        """
        Params:
        -------
        s (2D ndarray)
            state either {0,1} or {+/-1}
        params (ndarray)
            (h,J) vector
        """
        e = -fast_sum(params[s.shape[1]:],s)
        e -= np.dot(s,params[:s.shape[1]])
        return e

    @jit
    def mch_approximation( samples, dlamda ):
        dE = calc_e(samples,dlamda)
        dE -= dE.min()
        ZFraction = 1. / np.mean(np.exp(-dE))
        predsisj=calc_sisj( samples, weighted=np.exp(-dE)/len(dE),concat=True ) * ZFraction  
        if np.any(predsisj<-1.00000001) or np.any(predsisj>1.000000001):
            print(predsisj.min(),predsisj.max())
            raise Exception("Predicted values are beyond limits.")
        return predsisj
    return calc_e,mch_approximation


