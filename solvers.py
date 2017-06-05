# Module for class-based solvers for different Inverse Ising methods.
from __future__ import division
import entropy.entropy as entropy
from scipy.optimize import minimize
from multiprocessing import Pool,Array,Queue,Process
from misc.utils import unique_rows
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
        For exact: lambda params: return observables
    multipliers (ndarray=None)
    nWorkers (int=None)

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
    def __init__(self, n,
                 constraints=None,
                 calc_e=None,
                 calc_de=None,
                 calc_observables=None,
                 adj=None,
                 multipliers=None,
                 nWorkers=0):
        # Do basic checks on the inputs.
        assert type(n) is int
        assert (not calc_e is None), "Must define calc_e()."
        
        self.n = n
        self.constraints = constraints
        self.multipliers = multipliers
        
        self.calc_e = calc_e
        self.calc_de = calc_de
        self.calc_observables = calc_observables
        self.adj = adj
        
        self.nWorkers = nWorkers

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
        super(Exact,self).__init__(*args,**kwargs)
        if self.multipliers is None:
            self.multipliers = np.zeros(self.constraints.shape)

    def solve(self,
              initial_guess=None,
              tol=None,
              tolNorm=None,
              disp=False,
              max_param_value=50,
              fsolve_kwargs={}):
        """
        Params:
        ------
        initial_guess (ndarray=None)
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
        Output from scipy.optimize.minimize
        """
        if not initial_guess is None:
            assert len(initial_guess)==len(self.multipliers)
            self.multipliers = initial_guess.copy()
        else: initial_guess = np.zeros((len(self.multipliers)))
        
        def f(params):
            if np.any(np.abs(params)>max_param_value):
                return [1e30]*len(params)
            return np.linalg.norm( self.calc_observables(params)-self.constraints )

        return minimize(f,initial_guess,**fsolve_kwargs)

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
# End Exact



def unwrap_self_worker_obj(arg, **kwarg):
    return MPF.worker_objective_task(*arg, **kwarg)

class MPF(Solver):
    def __init__(self, *args, **kwargs):
        """
        Parallelized implementation of Minimum Probability Flow algorithm.
        Slowest step is the computation of the energy of a given state. Make this as fast as possible.

        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda=None)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        nWorkers (int=0)
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        
        Attributes:
        -----------
        
        Methods:
        --------
        """
        super(MPF,self).__init__(*args,**kwargs)
        self.adj = adj
        if self.nWorkers>0:
            self.solve = self.solve_parallel
        
    @staticmethod
    def worker_objective_task( s, Xcount, adjacentStates, params, calc_e ):
        return Xcount * np.sum(np.exp( .5*(calc_e(s[None,:],params) 
                                           - calc_e(adjacentStates,params) ) ))
 
    def K( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute objective function.
        
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

        Returns:
        --------
        logK (float)
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
    # End logK

    def solve( self, X, 
               initial_guess=None,
               method='L-BFGS-B',
               all_connected=True,
               parameter_limits=100,
               solver_kwargs={'maxiter':100,'disp':True,'ftol':1e-15},
               uselog=True,
               ):
        """
        Minimize MPF objective function using scipy.optimize.minimize.

        Params:
        -------
        X (ndata x ndim ndarray)
            array of states compatible with given energy and adjacent neighbors functions
        adj (lambda state)
            returns adjacent states for any given state
        all_connected (bool=True)
            switch for summing over all states that data sets could be connected to or just summing over
            non-data states (second summation in Eq 10 in Sohl-Dickstein 2011)
        iterate (int=0)
            number of times to try new initial conditions if first try doesn't work. Right now, this is a
            pretty coarse test because the fit can be good even without converging.
        parameter_limits (float)
            some limit to constrain the space that the solver has to search. This is the maximum allowed
            magnitude of any single parameter.
        solver_kwargs (dict)
            For scipy.optimize.minimize.

        Returns:
        --------
        soln (ndarray)
            found solution to problem
        output (dict)
            full output from minimize solver
        """
        assert parameter_limits>0

        # Convert from {0,1} to {+/-1} asis.
        X = (X+1)/2
        
        if not self.calc_de is None:
            includeGrad = True
        else:
            includeGrad = False
        X = X.astype(float)
        if initial_guess is None:
            initial_guess = entropy.calc_sisj( X, concat=True )
   
        # Get list of unique data states and how frequently they appear.
        Xuniq = X[unique_rows(X)]
        ix = unique_rows(X,return_inverse=True)
        Xcount = np.bincount(ix)
        M,N = Xuniq.shape
        
        adjacentStates = []
        for s in Xuniq:
            adjacentStates.append( self.adj(s) )
            # Remove states already in data.
            if not all_connected:
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
        soln = minimize( f, initial_guess,
                         bounds=[(-parameter_limits,parameter_limits)]*len(initial_guess),
                         method=method, jac=includeGrad, options=solver_kwargs )
        # NOTE: Returning soln details in terms of {0,1} basis.
        return convert_params(soln['x'][:self.n],soln['x'][self.n:],'11',True), soln
# End MPFSolver



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
