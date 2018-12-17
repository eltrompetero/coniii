# =============================================================================================== #
# ConIII module for algorithms for solving the inverse Ising problem.
# Authors: Edward Lee (edlee@alumni.princeton.edu) and Bryan Daniels (bryan.daniels.1@asu.edu)
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
# =============================================================================================== #
from scipy.optimize import minimize,fmin_ncg,minimize_scalar
import multiprocess as mp
import copy
from . import mean_field_ising
from warnings import warn
from .utils import *
from .samplers import *


class Solver():
    """
    Base class for declaring common methods and attributes for inverse maxent algorithms.

    Necessary members to define
    ---------------------------
    calc_e : lambda function
        Takes states and parameters to calculate the energies.
    calc_observables : lambda function
        Calculate observables from given sample of states.
        lambda X: Y
        where X is of dimensions (n_samples, n_dim)
        and Y is of dimensions (n_samples, n_constraints)

    Methods to customize
    --------------------
    solve
    """
    def __init__(self, n,
                 calc_de=None,
                 calc_observables=None,
                 calc_observables_multipliers=None,
                 adj=None,
                 multipliers=None,
                 constraints=None,
                 sample_size=None,
                 sample_method=None,
                 mch_approximation=None,
                 n_cpus=None,
                 rng=None,
                 verbose=False):
        """
        Parameters
        ----------
        n : int
            System size given by number of spins.
        calc_de : function, None
            Function for calculating derivative of energy with respect to the parameters. Takes in 2d
            state array and index of the parameter.
            Defn: lambda state_2d,ix : delta_energy
        calc_observables : function, None
            Defn: lambda params : observables
        calc_observables_multipliers : function, None
            Calculate predicted observables using the parameters.
            Defn: lambda parameters : pred_observables
        adj : function, None
            Return adjacency matrix.
        multipliers : ndarray, None
            Langrangian multipliers or parameters.
        constraints : ndarray, None
            Correlations to constrain.
        sample_size : int, None
        sample_method : str, None
        n_cpus : int, None
            Number of cores to use for parallelized code. If this is set to 0,  sequential sampler
            will be used. This should be set if multiprocess module does not work.
        verbose : bool, False
        """

        # Basic checks on the inputs.
        assert type(n) is int
        if not sample_size is None:
            assert type(sample_size) is int
        if not n_cpus is None:
            assert type(n_cpus) is int
        
        self.n = n
        self.multipliers = multipliers
        self.constraints = constraints
        self.sampleSize = sample_size
        self.sampleMethod = sample_method
        self.mch_approximation = mch_approximation
        
        self.calc_observables = calc_observables
        self.calc_observables_multipliers = calc_observables_multipliers
        self.calc_e = lambda s, multipliers : -self.calc_observables(s).dot(multipliers)
        self.calc_de = calc_de
        self.adj = adj
        
        self.rng = rng or np.random.RandomState()  # this will get passed to sampler if it is set up
        self.nCpus = n_cpus or mp.cpu_count()-1
        self.verbose = verbose

    def solve(self):
        return
              
    def estimate_jac(self, eps=1e-3):
        return 

    def setup_sampler(self,
                      sample_method='metropolis',
                      sampler_kwargs={}):
        """
        Instantiate sampler class object. Uses self.rng as the random number generator.

        Parameters
        ----------
        sample_method : str, 'metropolis'
            'metropolis'
        sampler_kwargs : dict, {}
            Kwargs that can be passed into the initialization function for the sampler.
        """

        sample_method = sample_method or self.sampleMethod
        
        if sample_method=='metropolis':
            self.sampleMethod = sample_method
            self.sampler = Metropolis( self.n, self.multipliers, self.calc_e,
                                       n_cpus=self.nCpus,
                                       rng=self.rng,
                                       **sampler_kwargs )
      
        elif sample_method=='ising_metropolis':
            raise NotImplementedError("FastMCIsing is no longer available.")
        else:
           raise NotImplementedError("Unrecognized sampler %s."%sample_method)
        self.samples = None

    def generate_samples(self, n_iters, burn_in,
                         multipliers=None,
                         sample_size=None,
                         sample_method=None,
                         generate_kwargs={}):
        """
        Wrapper around generate_samples() generate_samples_parallel() methods in samplers.

        Samples are saved to self.samples.

        Parameters
        ----------
        n_iters : int
        burn_in : int 
            Burn in is handled automatically in REMC.
        multipliers : ndarray,None
        sample_size : int,None
        sample_method : str,None
        generate_kwargs : dict,{}
        """

        assert not (self.sampler is None), "Must call setup_sampler() first."
        
        if multipliers is None:
            multipliers = self.multipliers
        sample_method = sample_method or self.sampleMethod
        sample_size = sample_size or self.sampleSize
        
        # When sequential sampling should be used.
        if self.nCpus<=1:
            if sample_method=='metropolis':
                self.sampler.theta = multipliers.copy()
                # Burn in.
                self.sampler.generate_samples(sample_size,
                                              n_iters=burn_in)
                self.sampler.generate_samples(sample_size,
                                              n_iters=n_iters)
                self.samples = self.sampler.samples

            else:
               raise NotImplementedError("Unrecognized sampler.")
        # When parallel sampling using the multiprocess module.
        else:
            if sample_method=='metropolis':
                self.sampler.theta = multipliers.copy()
                self.sampler.generate_samples_parallel(sample_size,
                                                       n_iters=burn_in+n_iters)
                self.samples = self.sampler.samples

            else:
               raise NotImplementedError("Unrecognized sampler.")
# end Solver


class Enumerate(Solver):
    """
    Class for solving +/-1 symmetric Ising model maxent problems by gradient descent with flexibility to put
    in arbitrary constraints.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        n : int
            System size.
        calc_observables_multipliers : function
            Function for calculating the observables given a set of multipliers. Function call is 
            lambda params: return observables
        calc_observables : function
            lambda params: return observables
        """
        super(Enumerate, self).__init__(*args, **kwargs)

    def solve(self,
              constraints=None,
              samples=None,
              initial_guess=None,
              max_param_value=50,
              full_output=False,
              fsolve_kwargs={'method':'powell'}):
        """
        Must specify either constraints (the correlations) or samples from which the
        correlations will be calculated using self.calc_observables.

        Parameters
        ----------
        constraints : ndarray, None
            Correlations that will be fit to.
        samples : ndarray, None
            (n_samples, n_dim)
        initial_guess : ndarray, None
            initial starting point
        max_param_value : float, 50
            Absolute value of max parameter value. Bounds can also be set in the kwargs
            passed to the minimizer, in which case this should be set to None.
        full_output : bool, False
            If True, return output from scipy.optimize.minimize.
        fsolve_kwargs : dict, {'method':'powell'}
            Powell method is slower but tends to converge better.

        Returns
        -------
        ndarray
            Solved multipliers (parameters).
        dict, optional
            Output from scipy.optimize.minimize.
        """

        if not constraints is None:
            self.constraints = constraints
        elif not samples is None:
            self.constraints = self.calc_observables(samples).mean(0)
        else:
            raise Exception("Must specify either constraints or samples.")
        
        if not initial_guess is None:
            self.multipliers = initial_guess.copy()
        else: initial_guess = np.zeros((len(self.constraints)))
        
        if not max_param_value is None:
            def f(params):
                if np.any(np.abs(params)>max_param_value):
                    return 1e30
                return np.linalg.norm( self.calc_observables_multipliers(params)-self.constraints )
        else:
            def f(params):
                return np.linalg.norm( self.calc_observables_multipliers(params)-self.constraints )

        soln = minimize(f, initial_guess, **fsolve_kwargs)
        self.multipliers = soln['x']
        if full_output:
            return soln['x'], soln
        return soln['x']
#end Enumerate


def unwrap_self_worker_obj(arg, **kwarg):
    return MPF.worker_objective_task(*arg, **kwarg)

class MPF(Solver):
    def __init__(self, *args, **kwargs):
        """
        Parallelized implementation of Minimum Probability Flow algorithm.
        Slowest step is the computation of the energy of a given state. Make this as fast as possible.

        Parameters
        ----------
        n : int
            System size.
        adj : function, None
            function for getting all the neighbors of any given state
        calc_de : function, None
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        n_cpus : int, 0
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        """

        super(MPF,self).__init__(*args,**kwargs)
        
    @staticmethod
    def worker_objective_task( s, Xcount, adjacentStates, params, calc_e ):
        return Xcount * np.sum(np.exp( .5*(calc_e(s[None,:],params) 
                                           - calc_e(adjacentStates,params) ) ))
 
    def K( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute objective function.
        
        Parameters
        ----------
        Xuniq : ndarray
            (ndata x ndims)
            unique states that appear in the data
        Xcount : ndarray of int
            number of times that each unique state appears in the data
        adjacentStates : list of ndarray
            list of adjacent states for each given unique state
        params : ndarray
            parameters for computation of energy

        Returns
        -------
        K : float
        """

        obj = 0.
        objGrad = np.zeros((params.size))
        for i,s in enumerate(Xuniq):
            dobj = Xcount[i] * np.exp( .5*(self.calc_e(s[None,:], params) 
                                           - self.calc_e(adjacentStates[i], params) ) )
            if not self.calc_de is None:
                for j in range(params.size):
                    if dobj.size != adjacentStates[i].shape[0]:
                        raise Exception("Sizes do not match")
                    objGrad[j] += .5 * (dobj * ( self.calc_de(s[None,:],j) 
                                        - self.calc_de(adjacentStates[i],j) )).sum()
            obj += dobj.sum()
        #else:
        #    # Parallel loop through objective function calculation for each state in the data.
        #    obj = [self.pool.apply( unwrap_self_worker_obj, 
        #                            args=([Xuniq[i],Xcount[i],adjacentStates[i],params,self.calc_e],) ) 
        #                for i in range(Xuniq.shape[0])]
        #    obj = obj.sum()

        #    if not self.calc_de is None:
        #        from warning import warn
        #        warn("Gradient computation not written for parallel loop.")

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
        
        Parameters
        ----------
        Xuniq : ndarray
            (n_samples, n_dim)
            unique states that appear in the data
        Xcount : ndarray of int
            number of times that each unique state appears in the data
        adjacentStates : list of ndarray
            list of adjacent states for each given unique state
        params : ndarray
            parameters for computation of energy

        Returns
        -------
        logK : float
        """

        from scipy.special import logsumexp

        obj = 0.
        objGrad = np.zeros((params.size))
        power=np.zeros((len(Xuniq), len(adjacentStates[0])))  # energy differences
        for i,s in enumerate(Xuniq):
            power[i,:] = .5*( self.calc_e(s[None,:],params) - self.calc_e(adjacentStates[i],params) )
            
        obj = logsumexp( power + np.log(Xcount)[:,None] -np.log(Xcount.sum()) )
        
        if self.calc_de is None:
            return obj

        # coefficients that come out from taking derivative of exp
        for i in range(params.size):
            gradcoef = np.zeros((len(Xuniq), len(adjacentStates[0])))  
            for j,s in enumerate(Xuniq): 
                gradcoef[j,:] = .5 * ( self.calc_de(s[None,:],i) - self.calc_de(adjacentStates[j],i) )
            power -= power.max()
            objGrad[i] = ((gradcoef*np.exp(power)*Xcount[:,None]).sum() /
                          (np.exp(power)*Xcount[:,None]).sum())
        objGrad -= np.log(Xcount.sum())
        
        if objGrad.size==1:
            raise Exception("")
        return obj, objGrad

    def list_adjacent_states(self, Xuniq, all_connected):
        """
        Use self.adj to evaluate all adjacent states in Xuniq.

        Parameters
        ----------
        Xuniq : ndarray
        all_connected : bool

        Returns
        -------
        adjacentStates
        """

        adjacentStates = []
        for s in Xuniq:
            adjacentStates.append( self.adj(s) )
            # Remove states already in data
            if not all_connected:
                ix = np.zeros((s.size))==0
                for i,t in enumerate(adjacentStates[-1]):
                    if np.any(np.all(t[None,:]==Xuniq,1)):
                        ix[i] = False
                if np.sum(ix)==X.shape[1]:
                    raise Exception("This data set does not satisfy MPF assumption that each \
                                    state be connected to at least one non-data state (?)")
                adjacentStates[-1] = adjacentStates[-1][ix]
        return adjacentStates

    def solve(self,
              X=None, 
              initial_guess=None,
              method='L-BFGS-B',
              full_output=False,
              all_connected=True,
              parameter_limits=100,
              solver_kwargs={'maxiter':100,'disp':False,'ftol':1e-15},
              uselog=True):
        """
        Minimize MPF objective function using scipy.optimize.minimize.

        Parameters
        ----------
        X : ndarray
            (ndata, ndim)
            array of states compatible with given energy and adjacent neighbors functions
        adj : lambda state
            returns adjacent states for any given state
        all_connected : bool,True
            switch for summing over all states that data sets could be connected to or just summing over
            non-data states (second summation in Eq 10 in Sohl-Dickstein 2011)
        iterate : int,0
            number of times to try new initial conditions if first try doesn't work. Right now, this is a
            pretty coarse test because the fit can be good even without converging.
        parameter_limits : float
            some limit to constrain the space that the solver has to search. This is the maximum allowed
            magnitude of any single parameter.
        solver_kwargs : dict
            For scipy.optimize.minimize.
        uselog : bool,True
            If True, calculate log of the objective function. This can help with numerical precision
            errors.

        Returns
        -------
        soln : ndarray
            found solution to problem
        output : dict
            full output from minimize solver
        """
        
        from .utils import split_concat_params
        assert parameter_limits>0
        assert not X is None, "samples from distribution of states must be provided for MPF"

        # Convert from {0,1} to {+/-1} asis.
        X = (X+1)/2
        
        if not self.calc_de is None:
            includeGrad = True
        else:
            includeGrad = False
        X = X.astype(float)
        if initial_guess is None:
            initial_guess = self.calc_observables(X).mean(0)
         
        # Get list of unique data states and how frequently they appear.
        Xuniq = X[unique_rows(X)]
        ix = unique_rows(X, return_inverse=True)
        Xcount = np.bincount(ix)
        adjacentStates = self.list_adjacent_states(Xuniq, all_connected)

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
        self.multipliers = soln['x']
        if full_output:
            return ising_convert_params( split_concat_params(soln['x'], self.n), '11', True), soln
        return ising_convert_params( split_concat_params(soln['x'], self.n), '11', True)
#end MPF


class MCH(Solver):
    """Class for solving maxent problems using the Monte Carlo Histogram method.

    Broderick, T., Dudik, M., Tkacik, G., Schapire, R. E. & Bialek, W. Faster solutions of the
    inverse pairwise Ising problem. arXiv 1-8 (2007).
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        calc_observables : function
            takes in samples as argument
        sample_method : str
            Can be 'metropolis'.
        sample_size : int
            Number of samples to use MCH sampling step.
        mch_approximation : function
            For performing the MCH approximation step. Is specific to the maxent model. For the
            pairwise Ising model, this can be defined by using
            `coniii.utils.define_ising_helper_functions()`.
        n_cpus : int
            If 1 or less no parallel processing, other numbers above 0 specify number of cores to use.
        """

        super(MCH, self).__init__(*args, **kwargs)
        assert not self.sampleSize is None, "Must specify sample_size."
        assert not self.mch_approximation is None, "Must specify mch_approximation."
        assert not self.calc_observables is None, "Must specify calc_observables."
        
        # Sampling parameters.
        self.sampler = None
        self.samples = None
        
        self.setup_sampler()
    
    def solve(self,
              initial_guess=None,
              constraints=None,
              X=None,
              tol=None,
              tolNorm=None,
              n_iters=30,
              burn_in=30,
              maxiter=10,
              custom_convergence_f=None,
              iprint=False,
              full_output=False,
              learn_params_kwargs={'maxdlamda':1, 'eta':1},
              generate_kwargs={},
              **kwargs):
        """
        Solve for maxent model parameters using MCH routine.
        
        Parameters
        ----------
        initial_guess : ndarray, None
            Initial starting point.
        constraints : ndarray, None
            Vector of correlations to fit.
        X : ndarray, None
            If instead of constraints, you wish to pass the raw data on which to calculate the
            constraints using self.calc_observables.
        tol : float, None
            Maximum error allowed in any observable.
        tolNorm : float, None
            Norm error allowed in found solution.
        n_iters : int, 30
            Number of iterations to make between samples in MCMC sampling.
        burn_in : int, 30
            Initial burn in from random sample when MC sampling.
        max_iter : int, 10
            Max number of iterations of MC sampling and MCH approximation.
        custom_convergence_f : function, None
            Function for determining convergence criterion. At each iteration, this function should
            return the next set of learn_params_kwargs and optionally the sample size.

            As an example:
	    def learn_settings(i):
		'''
		Take in the iteration counter and set the maximum change allowed in any given 
		parameter (maxdlamda) and the multiplicative factor eta, where 
		d(parameter) = (error in observable) * eta.
		
		Additional option is to also return the sample size for that step by returning a 
		tuple. Larger sample sizes are necessary for higher accuracy.
		'''
		if i<10:
		    return {'maxdlamda':1,'eta':1}
		else:
		    return {'maxdlamda':.05,'eta':.05}
        iprint : bool, False
        full_output : bool, False
            If True, also return the errflag and error history.
        learn_parameters_kwargs : dict, {'maxdlamda':1,'eta':1}
        generate_kwargs : dict, {}

        Returns
        -------
        ndarray
            Found solution to inverse problem.
        int
            Error flag.
            0, converged within given criterion
            1, max iterations reached
        ndarray
            Log of errors in matching constraints at each step of iteration.
        """

        if 'disp' in kwargs.keys():
            raise Exception("disp kwarg has been replaced with iprint.")
        if 'burnin' in kwargs.keys():
            warn("burnin kwarg has been replaced with burn_in.")
            burn_in = kwargs['burnin']
        if (self.n*10)>burn_in:
            warn("Number of burn in MCMC iterations between samples may be too small for "+
                 "convergence to stationary distribution.")
        if (self.n*10)>n_iters:
            warn("Number of MCMC iterations between samples may be too small for convergence to "+
                 "stationary distribution.")

        errors = []  # history of errors to track

        # Read in constraints.
        if not constraints is None:
            self.constraints = constraints
        elif not X is None:
            self.constraints = self.calc_observables(X).mean(0)
        else: assert not self.constraints is None
        
        # Set initial guess for parameters. self._multipliers is where the current guess for the
        # parameters is stored.
        if not (initial_guess is None):
            assert len(initial_guess)==len(self.constraints)
            self._multipliers = initial_guess.copy()
        else:
            self._multipliers = np.zeros((len(self.constraints)))
        tol = tol or 1/np.sqrt(self.sampleSize)
        tolNorm = tolNorm or np.sqrt( 1/self.sampleSize )*len(self._multipliers)
        
        # Redefine function for automatically adjusting learn_params_kwargs so that it returns the
        # MCH iterator settings and the sample size if it doesn't already.
        if custom_convergence_f is None:
            custom_convergence_f = lambda i:learn_params_kwargs,self.sampleSize
        if type(custom_convergence_f(0)) is dict:
            custom_convergence_f_ = custom_convergence_f
            custom_convergence_f = lambda i:(custom_convergence_f_(i),self.sampleSize)
        assert 'maxdlamda' and 'eta' in list(custom_convergence_f(0)[0].keys())
        assert type(custom_convergence_f(0)[1]) is int
        
        
        # Generate initial set of samples.
        self.generate_samples( n_iters,burn_in,
                               multipliers=self._multipliers,
                               generate_kwargs=generate_kwargs )
        thisConstraints = self.calc_observables(self.samples).mean(0)
        errors.append( thisConstraints-self.constraints )
        if iprint=='detailed': print(self._multipliers)


        # MCH iterations.
        counter = 0  # number of MCMC and MCH steps
        keepLooping = True  # loop control
        learn_params_kwargs,self.sampleSize = custom_convergence_f(counter)
        while keepLooping:
            # MCH step
            if iprint:
                print("Iterating parameters with MCH...")
            self.learn_parameters_mch(thisConstraints,**learn_params_kwargs)
            if iprint=='detailed':
                print("After MCH step, the parameters are...")
                print(self._multipliers)
            
            # MC sampling step
            if iprint:
                print("Sampling...")
            self.generate_samples( n_iters,burn_in,
                                   multipliers=self._multipliers,
                                   generate_kwargs=generate_kwargs )
            thisConstraints = self.calc_observables(self.samples).mean(0)
            counter += 1
            
            errors.append( thisConstraints-self.constraints )
            if iprint=='detailed':
                print("Error is %1.4f"%np.linalg.norm(errors[-1]))
            # Exit criteria.
            if ( np.linalg.norm(errors[-1])<tolNorm
                 and np.all(np.abs(thisConstraints-self.constraints)<tol) ):
                if iprint: print("Solved.")
                errflag=0
                keepLooping=False
            elif counter>maxiter:
                if iprint: print("Over maxiter")
                errflag=1
                keepLooping=False
            else:
                learn_params_kwargs, self.sampleSize = custom_convergence_f(counter)
        
        self.multipliers = self._multipliers.copy()
        if full_output:
            return self.multipliers, errflag, np.vstack((errors))
        return self.multipliers

    def estimate_jac(self, eps=1e-3):
        """
        Approximation Jacobian using the MCH approximation. 

        Parameters
        ----------
        eps : float,1e-3

        Returns
        -------
        jac : ndarray
            Jacobian is an n x n matrix where each row corresponds to the behavior of fvec wrt to a
            single parameter.
        """

        dlamda = np.zeros(self._multipliers.shape)
        jac = np.zeros((self._multipliers.size,self._multipliers.size))
        print("evaluating jac")
        for i in range(len(self._multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac

    def learn_parameters_mch(self,
                             estConstraints,
                             maxdlamda=1,
                             maxdlamdaNorm=1, 
                             maxLearningSteps=50,
                             eta=1 ):
        """
        Parameters
        ----------
        estConstraints : ndarray
            Constraints estimated from MCH approximation.
        maxdlamda : float, 1
            Max allowed magnitude for any element of dlamda vector before exiting.
        maxdlamdaNorm : float, 1
            Max allowed norm of dlamda vector before exiting.
        maxLearningSteps : int
            max learning steps before ending MCH
        eta : float, 1
            factor for changing dlamda

        Returns
        -------
        ndarray
            MCH estimate for constraints from parameters lamda+dlamda.
        """

        keepLearning = True
        dlamda = np.zeros((self.constraints.size))
        learningSteps = 0
        distance = 1
        
        while keepLearning:
            # Get change in parameters.
            # If observable is too large, then corresponding energy term has to go down 
            # (think of double negative).
            dlamda += -(estConstraints-self.constraints) * np.min([distance,1.]) * eta
            #dMultipliers /= dMultipliers.max()
            
            # Predict distribution with new parameters.
            estConstraints = self.mch_approximation( self.samples, dlamda )
            distance = np.linalg.norm( estConstraints-self.constraints )
                        
            # Counter.
            learningSteps += 1

            # Evaluate exit criteria.
            if np.linalg.norm(dlamda)>maxdlamdaNorm or np.any(np.abs(dlamda)>maxdlamda):
                keepLearning = False
            elif learningSteps>maxLearningSteps:
                keepLearning = False

        self._multipliers += dlamda
        return estConstraints
#end MCH


class MCHIncompleteData(MCH):
    """
    Class for solving maxent problems using the Monte Carlo Histogram method on
    incomplete data where some spins may not be visible.

    Broderick, T., Dudik, M., Tkacik, G., Schapire, R. E. & Bialek, W. Faster
    solutions of the inverse pairwise Ising problem. arXiv 1-8 (2007).

    NOTE: This only works for Ising model.
          Not ready for release.
    """
    def __init__(self, *args, **kwargs):
        """
        Not ready for release.
        """

        warn("MCHIncompleteData is not officially released as part of ConIII.")
        super(MCHIncompleteData,self).__init__(*args,**kwargs)
        self.condSamples = []
        
    def solve(self,
              X=None,
              constraints=None,
              initial_guess=None,
              cond_sample_size=100,
              cond_sample_iters=100,
              tol=None,
              tolNorm=None,
              n_iters=30,
              burn_in=30,
              maxiter=10,
              disp=False,
              full_output=False,
              learn_params_kwargs={},
              generate_kwargs={}):
        """
        Solve for parameters using MCH routine.
        
        Parameters
        ----------
        X                       : ndarray
        constraints             : ndarray
            Constraints calculated from the incomplete data (accounting for missing data points).
        initial_guess           : ndarray=None
            initial starting point
        cond_sample_size        : int or function
            Number of samples to make for conditional distribution.
            If function is passed in, it will be passed number of missing spins and must return an int.
        cond_sample_iters       : int or function
            Number of MC iterations to make between samples.
        tol                     : float=None
            maximum error allowed in any observable
        tolNorm                 : float
            norm error allowed in found solution
        n_iters                 : int=30
            Number of iterations to make between samples in MCMC sampling.
        burn_in (int=30)
        disp                    : int=0
            0, no output
            1, some detail
            2, most detail
        full_output : bool,False
            Return errflag and errors at each iteration if True.
        learn_parameters_kwargs : dict
        generate_kwargs         : dict

        Returns
        -------
        parameters : ndarray
            Found solution.
        errflag : int
        errors : ndarray
            Errors in matching constraints at each step of iteration.
        """

        # Check args.
        import types
        assert (not X is None) and (not constraints is None), "Must provide data and constriants."
        self.constraints = constraints
        if type(cond_sample_size) is int:
            f_cond_sample_size = lambda n: cond_sample_size
        elif type(cond_sample_size) is types.FunctionType:
            f_cond_sample_size = cond_sample_size 
        if type(cond_sample_iters) is int:
            f_cond_sample_iters = lambda n: cond_sample_iters
        elif type(cond_sample_iters) is types.FunctionType:
            f_cond_sample_iters = cond_sample_iters 

        # Set initial guess for parameters.
        if not (initial_guess is None):
            assert len(initial_guess)==len(self.constraints)
            self._multipliers = initial_guess.copy()
        else:
            self._multipliers = np.zeros((len(self.constraints)))
        tol = tol or 1/np.sqrt(self.sampleSize)
        tolNorm = tolNorm or np.sqrt( 1/self.sampleSize )*len(self._multipliers)
        errors = []  # history of errors to track

        # Get unique incomplete data points.
        incompleteIx = (X==0).any(1)
        uIncompleteStates = X[incompleteIx][unique_rows(X[incompleteIx])]
        # Frequency of each unique state.
        uIncompleteStatesCount = np.bincount( unique_rows(X[incompleteIx],
                                                          return_inverse=True) )
        fullFraction = (len(X)-incompleteIx.sum())/len(X)
        if disp:
            print("There are %d unique states."%len(uIncompleteStatesCount))
        
        # Sample.
        if disp:
            print("Sampling...")
        self.generate_samples(n_iters,burn_in,
                              uIncompleteStates,f_cond_sample_size,f_cond_sample_iters,
                              generate_kwargs=generate_kwargs,disp=disp)
        thisConstraints = self.calc_observables(self.samples).mean(0)
        errors.append( thisConstraints-self.constraints )
        
        # MCH iterations.
        counter = 0
        keepLoop = True
        if disp>=2: print(self._multipliers)
        while keepLoop:
            if disp:
                print("Iterating parameters with MCH...")
            self.learn_parameters_mch(thisConstraints,
                                      fullFraction,
                                      uIncompleteStates,
                                      uIncompleteStatesCount,
                                      **learn_params_kwargs)
            if disp>=2:
                print("After MCH step, the parameters are...")
                print(self._multipliers)

            # Sample.
            if disp:
                print("Sampling...")
            self.generate_samples(n_iters,burn_in,
                                  uIncompleteStates,f_cond_sample_size,f_cond_sample_iters,
                                  generate_kwargs=generate_kwargs,disp=disp)

            thisConstraints = self.calc_observables(self.samples).mean(0)
            counter += 1
            
            # Exit criteria.
            errors.append( thisConstraints-self.constraints )
            if ( np.linalg.norm(errors[-1])<tolNorm
                 and np.all(np.abs(thisConstraints-self.constraints)<tol) ):
                print("Solved.")
                errflag=0
                keepLoop=False
            elif counter>maxiter:
                print("Over maxiter")
                errflag=1
                keepLoop=False
        
        self.multipliers = self._multipliers.copy()
        if full_output:
            return self.multipliers,errflag,np.vstack((errors))
        return self.multipliers

    def learn_parameters_mch(self,
                             estConstraints,
                             fullFraction,
                             uIncompleteStates,
                             uIncompleteStatesCount,
                             maxdlamda=1,
                             maxdlamdaNorm=1, 
                             maxLearningSteps=50,
                             eta=1 ):
        """
        Update parameters with MCH step. Update is proportional to the difference between the
        observables and the predicted observables after a small change to the parameters. This is
        calculated from likelihood maximization, and for the incomplete data points this corresponds
        to the marginal probability distribution weighted with the number of corresponding data
        points.

        Parameters
        ----------
        estConstraints : ndarray
        fullFraction : float
            Fraction of data points that are complete.
        uIncompleteStates : list-like
            Unique incomplete states in data.
        uIncompleteStatesCount : list-like
            Frequency of each unique data point.
        maxdlamda : float,1
        maxdlamdaNorm : float,1
        maxLearningSteps : int
            max learning steps before ending MCH
        eta : float,1
            factor for changing dlamda

        Returns
        -------
        estimatedConstraints : ndarray
        """

        keepLearning = True
        dlamda = np.zeros((self.constraints.size))
        learningSteps = 0
        distance = 1
        
        # for each data point, estimate the value of the observables with MCH
        # take the average of the predictions
        # minimize the diff btwn that avg and the goal
        while keepLearning:
            # Get change in parameters.
            # If observable is too large, then corresponding energy term has to go down 
            # (think of double negative).
            dlamda += -(estConstraints-self.constraints) * np.min([distance,1.]) * eta
            #dMultipliers /= dMultipliers.max()
            
            # Predict distribution with new parameters.
            # MCH approximation with complete data points.
            if fullFraction>0:
                estConstraints = self.mch_approximation( self.samples, dlamda ) * fullFraction
            else:
                estConstraints = np.zeros_like(dlamda)
            # MCH approximation with incomplete data points. These will contribute to the likelihood
            # by the fraction of data points they constitute. So, the total weight per data point is
            # p(incomplete)*p(state|incomplete)
            for i,s in enumerate(self.condSamples):
                estConstraints += ( (1-fullFraction)*
                                    (uIncompleteStatesCount[i]/uIncompleteStatesCount.sum())*
                                    self.mch_approximation(s,dlamda) )
            distance = np.linalg.norm( estConstraints-self.constraints )
                        
            # Counter.
            learningSteps += 1

            # Evaluate exit criteria.
            if np.linalg.norm(dlamda)>maxdlamdaNorm or np.any(np.abs(dlamda)>maxdlamda):
                keepLearning = False
            elif learningSteps>maxLearningSteps:
                keepLearning = False

        self._multipliers += dlamda
        return estConstraints

    def generate_samples(self, n_iters, burn_in, 
                         uIncompleteStates=None,
                         f_cond_sample_size=None,
                         f_cond_sample_iters=None,
                         sample_size=None,
                         sample_method=None,
                         initial_sample=None,
                         run_regular_sampler=True,
                         run_cond_sampler=True,
                         disp=0,
                         generate_kwargs={}):
        """Wrapper around generate_samples_parallel() from available samplers.

        Parameters
        ----------
        n_iters : int
        burn_in : int 
            I think burn in is handled automatically in REMC.
        uIncompleteStates : list of unique states
        f_cond_sample_size : lambda function
            Given the number of hidden spins, return the number of samples to take.
        f_cond_sample_iters : lambda function
            Given the number of hidden spins, return the number of MC iterations to make.
        sample_size : int
        sample_method : str
        initial_sample : ndarray
        generate_kwargs : dict
        """

        from datetime import datetime  # for debugging
        assert not (self.sampler is None), "Must call setup_sampler() first."

        sample_method = sample_method or self.sampleMethod
        sample_size = sample_size or self.sampleSize
        if initial_sample is None and (not self.samples is None) and len(self.samples)==self.sampleSize:
            initial_sample = self.samples
        
        if sample_method=='metropolis':
            self.sampler.theta = self._multipliers
                
            # Generate samples from full distribution.
            if run_regular_sampler:
                # Burn in.
                self.sampler.generate_samples_parallel( sample_size,
                                                        n_iters=burn_in,
                                                        initial_sample=initial_sample )
                self.sampler.generate_samples_parallel( sample_size,
                                                        n_iters=n_iters,
                                                        initial_sample=self.sampler.samples )
                self.samples = self.sampler.samples
            if run_cond_sampler: 
                # Sample from conditional distribution p(s_unobserved|s_observed) where s_observed
                # are the spins with data for the incomplete data points.
                def f(args):
                    """Function for parallelizing sampling of conditional distributions."""
                    i,s = args
                    frozenSpins = list(zip(np.where(s!=0)[0],s[s!=0]))
                    
                    if disp:
                        start = datetime.now() 
                    sample,E = self.sampler.generate_cond_samples(f_cond_sample_size(self.n-len(frozenSpins)),
                                                          frozenSpins,
                                                          burn_in=f_cond_sample_iters(self.n-len(frozenSpins)),
                                                          parallel=False,
                                                          **generate_kwargs)
                    if disp:
                        print("Done sampling %d out of %d unique states in %1.1f s."%(i+1,
                                                                      len(uIncompleteStates),
                                                                      (datetime.now()-start).total_seconds()))
                    return sample

                # Parallel sampling of conditional distributions. 
                pool = mp.Pool(self.nCpus)
                self.condSamples = pool.map( f,list(zip(list(range(len(uIncompleteStates))),uIncompleteStates)) )
                pool.close()
        else:
           raise NotImplementedError("Unrecognized sampler.")
# End MCHIncompleteData


class Pseudo(Solver):
    """
    Pseudolikelihood approximation to solving the inverse Ising problem as described in Aurell and Ekeberg,
    PRL 108, 090201 (2012).
    """
    def __init__(self, *args, **kwargs):
        """
        For this technique, must specify how to calculate the energy specific to the conditional probability
        of spin r given the rest of the spins. These will be passed in with "get_observables_r" and
        "calc_observables_r".
        
        Parameters
        ----------
        get_multipliers_r : lambda function
            Takes index r and multipliers.
            Defn: lambda r,multipliers : r_multipliers
        calc_observables_r : lambda function
            Takes index r and samples X.
            Defn: lambda r,X : r_observable
        """

        self.calc_observables_r = kwargs.get('calc_observables_r',None)
        self.get_multipliers_r = kwargs.get('get_multipliers_r',None)
        assert not ( (self.calc_observables_r is None) or (self.get_multipliers_r is None) )
        del kwargs['calc_observables_r'],kwargs['get_multipliers_r']
        super(Pseudo,self).__init__(*args,**kwargs)

    def solve(self, *args, **kwargs):
        """
        Two different methods are implemented and can be called from self.solve. One is specific to the Ising
        model and the other uses a general all-purpose optimization (scipy.optimize) to solve the problem.

        Parameters
        ----------
        general_case : bool, True
            If True, uses self.calc_observables_r and self.get_multipliers_r to maximize the resulting
            pseudolikelihood (self._solve_general). Else an algorithm specific to the Ising model is
            implemented (self._solve_ising).

        Returns
        -------
        ndarray
            multipliers
        """

        if kwargs.get('general_case',False):
            del kwargs['general_case']
            return self._solve_general(*args,**kwargs)
        return self._solve_ising(*args,**kwargs)

    def _solve_general(self, X=None, initial_guess=None, return_all=False, solver_kwargs={}):
        """
        Solve for Langrangian parameters according to pseudolikelihood algorithm.

        Parameters
        ----------
        X : ndarray
            Data set if dimensions (n_samples, n_dim).
        initial_guess : ndarray, None
            Initial guess for the parameter values.
        return_all : bool, False
            If True, return output from scipy.minimize() routine.
        solver_kwargs : dict, {}
            kwargs for scipy.minimize().

        Returns
        -------
        ndarray
            multipliers
        dict
            Output from scipy.minimize.
        """

        def f(params):
            loglikelihood = 0
            for r in range(self.n):
                E = -self.calc_observables_r(r,X).dot(self.get_multipliers_r(r,params))
                loglikelihood += -np.log( 1+np.exp(2*E) ).sum() 
            return -loglikelihood
        
        soln = minimize(f,initial_guess,**solver_kwargs)
        self.multipliers = soln['x']
        if return_all:
            return soln['x'],soln
        return soln['x']

    def _solve_ising(self, X=None, initial_guess=None):
        """
        Solve Ising model specifically.

        Parameters
        ----------
        X : ndarray
            Data set if dimensions (n_samples, n_dim).
        initial_guess : ndarray
            Initial guess for the parameter values.

        Returns
        -------
        ndarray
            multipliers
        """

        X = (X + 1)/2  # change from {-1,1} to {0,1}
        
        # start at freq. model params?
        freqs = np.mean(X,axis=0)
        hList = -np.log(freqs/(1.-freqs))
        Jfinal = np.zeros((self.n,self.n))

        for r in range(self.n):
            Jr0 = np.zeros(self.n)
            Jr0[r] = hList[r]
            
            XRhat = X.copy()
            XRhat[:,r] = np.ones(len(X))
            # calculate once and pass to hessian algorithm for speed
            pairCoocRhat = self.pair_cooc_mat(XRhat)
            
            Lr = lambda Jr: - self.cond_log_likelihood(r,X,Jr)
            fprime = lambda Jr: self.cond_jac(r,X,Jr)
            fhess = lambda Jr: self.cond_hess(r,X,Jr,pairCoocRhat=pairCoocRhat)
            
            Jr = fmin_ncg(Lr, Jr0, fprime, fhess=fhess, disp=False)
            Jfinal[r] = Jr

        Jfinal = -0.5*( Jfinal + Jfinal.T )
        hfinal = Jfinal[np.diag_indices(self.n)]

        # Convert parameters into {-1,1} basis as is standard for this package.
        Jfinal[np.diag_indices(self.n)] = 0
        self.multipliers = convert_params( hfinal, squareform(Jfinal)*2, '11', concat=True )

        return self.multipliers

    def cond_log_likelihood(self, r, X, Jr):
        """
        Equals the conditional log likelihood -L_r.
        
        Parameters
        ----------
        r : int
            individual index
        X : ndarray
            binary matrix, (# X) x (dimension of system)
        Jr : ndarray
            (dimension of system) x (1)

        Returns
        -------
        float
        """

        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        invPs = 1. + np.exp( energies )
        logLs = np.log( invPs )

        return -logLs.sum()

    def cond_jac(self, r, X, Jr):
        """
        Returns d cond_log_likelihood / d Jr,
        with shape (dimension of system)
        """

        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        coocs = np.repeat([sigmaRtilde],self.n,axis=0).T * samplesRhat # (#X)x(self.n)

        return np.dot( coocs.T, 1./(1. + np.exp(-energies)) )

    def cond_hess(self, r, X, Jr, pairCoocRhat=None):
        """Returns d^2 cond_log_likelihood / d Jri d Jrj, with shape
        (dimension of system)x(dimension of system)

        Current implementation uses more memory for speed.
        For large sample size, it may make sense to break up differently
        if too much memory is being used.

        Parameters
        ----------
        pairCooc : ndarray, None
            Pass pair_cooc_mat(X) to speed calculation.
        """

        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        # pairCooc has shape (# X)x(n)x(n)
        if pairCoocRhat is None:
            pairCoocRhat = self.pair_cooc_mat(samplesRhat)
        
        energyMults = np.exp(-energies)/( (1.+np.exp(-energies))**2 ) # (# X)x(1)
        #filteredSigmaRtildeSq = filterVec * (2.*X[:,r] + 1.) # (# X)x(1)
        return np.dot( energyMults, pairCoocRhat )

    def pair_cooc_mat(self, X):
        """
        Returns matrix of shape (self.n)x(# X)x(self.n).
        
        For use with cond_hess.
        
        Slow because I haven't thought of a better way of doing it yet.
        """

        p = [ np.outer(f,f) for f in X ]
        return np.transpose(p,(1,0,2))

    def pseudo_log_likelhood(self, X, J):
        """
        (Could probably be made more efficient.)

        Parameters
        ----------
        X : ndarray
            binary matrix, (# of samples) x (dimension of system)
        J : ndarray
            (dimension of system) x (dimension of system)
            J should be symmetric
        """

        return np.sum([ cond_log_likelihood(r,X,J) \
                           for r in range(len(J)) ])
#end Pseudo


class ClusterExpansion(Solver):
    """
    Implementation of Adaptive Cluster Expansion for solving the inverse Ising problem, as
    described in John Barton and Simona Cocco, J. of Stat. Mech.  P03002 (2013).
    
    Specific to pairwise Ising constraints.
    """
    def __init__(self, *args, **kwargs):
        super(ClusterExpansion,self).__init__(*args,**kwargs)
        self.setup_sampler(kwargs.get('sample_method','metropolis'))
    
    def S(self, cluster, coocMat,
          deltaJdict={}, 
          useAnalyticResults=False,
          priorLmbda=0.,
          numSamples=None):
        """
        Calculate pairwise entropy of cluster.
        (First fits pairwise Ising model.)
        
        Parameters
        ----------
        cluster : list
            List of indices belonging to each cluster.
        coocMat : ndarray
            Pairwise correlations.
        deltaJdict : dict,{}
        useAnalyticResults : bool,False
            Probably want False until analytic formulas are changed to include prior on J

        Returns
        -------
        entropy : float
        Jfull : ndarray
            Matrix of couplings.
        """

        if len(cluster) == 0:
            raise Exception
        elif (len(cluster) == 1) and useAnalyticResults:
            p = coocMat[cluster[0],cluster[0]]
            J = np.array( [ [ -log( p / (1.-p) ) ] ] )
        elif (len(cluster) == 2) and useAnalyticResults:
            i = min(cluster[0],cluster[1])
            j = max(cluster[0],cluster[1])
            pi = coocMat[i,i]
            pj = coocMat[j,j]
            pij = coocMat[i,j]
            Jii1 = -log( pi / (1.-pi) )
            Jjj1 = -log( pj / (1.-pj) )
            Jii = -log( (pi - pij)/(1.-pi-pj+pij) )
            Jjj = -log( (pj - pij)/(1.-pi-pj+pij) )
            Jij = - log( pij ) + log( pi - pij ) + log( pj - pij )    \
                - log( 1.-pi-pj+pij )
            J = np.array( [ [ Jii, 0.5*Jij ], [ 0.5*Jij, Jjj ] ] )
        else:
            coocMatCluster = mean_field_ising.coocCluster(coocMat,cluster)
            Jinit = None # <--- potential for speed-up here
            J = mean_field_ising.findJmatrixAnalytic_CoocMat(coocMatCluster,
                                                             Jinit=Jinit,
                                                             priorLmbda=priorLmbda,
                                                             numSamples=numSamples)
        
        # make 'full' version of J (of size NxN)
        N = len(coocMat)
        Jfull = mean_field_ising.JfullFromCluster(J,cluster,N)
        
        ent = mean_field_ising.analyticEntropy(J)

        return ent, Jfull 

    def Sindependent(self, cluster, coocMat):
        """
        Entropy approximation assuming that each cluster appears independently of the others.

        Parameters
        ----------
        cluster : list
        coocMat : ndarray
            Pairwise correlations.

        Returns
        -------
        Sind : float
            Independent entropy.
        Jfull : ndarray
            Pairwise couplings.
        """
        
        # sort by cluster indices
        coocMatCluster = mean_field_ising.coocCluster(coocMat, cluster)
        # in case we're given an upper-triangular coocMat:
        coocMatCluster = mean_field_ising.symmetrizeUsingUpper(coocMatCluster)
        
        freqs = np.diag(coocMatCluster).copy()

        h = -np.log(freqs/(1.-freqs))
        Jind = np.diag(h)
        
        # independent approx
        Sinds = -freqs*np.log(freqs) - (1.-freqs)*np.log(1.-freqs)
        Sind = np.sum(Sinds)

        # make 'full' version of J (of size NfullxNfull)
        Nfull = len(coocMat)
        Jfull = mean_field_ising.JfullFromCluster(Jind, cluster, Nfull)

        return Sind, Jfull

    # "Algorithm 1"
    def deltaS(self, cluster, coocMat, 
               deltaSdict=None,
               deltaJdict=None,
               verbose=True,
               meanFieldRef=False,
               priorLmbda=0.,
               numSamples=None,
               independentRef=False,
               meanFieldPriorLmbda=None):
        """
        Parameters
        ----------
        cluster : list 
            List of indices in cluster
        coocMat : ndarray
        deltaSdict : dict,None
        deltaJdict : dict,None
        verbose : bool,True
        meanFieldRef : bool,False
        numSamples : int,None
        independentRef : bool,False
            If True, expand about independent entropy
        meanFieldRef : bool,False
            If True, expand about mean field entropy

        Returns
        -------
        deltaScluster
        deltaJcluster
        """

        if deltaSdict is None: deltaSdict = {}
        if deltaJdict is None: deltaJdict = {}
        
        if (independentRef and meanFieldRef) or \
           not (independentRef or meanFieldRef): raise Exception
        
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        
        cID = self.clusterID(cluster)
        if cID in deltaSdict:
            #print "deltaS: found answer for",cluster
            return deltaSdict[cID],deltaJdict[cID]
        elif verbose:
            print("deltaS: Calculating entropy for cluster",cluster)
        
        # start with full entropy (and J)
        deltaScluster,deltaJcluster = self.S(cluster,coocMat,
                                            deltaJdict,
                                            priorLmbda=priorLmbda,
                                            numSamples=numSamples)
        
        if independentRef:
            # subtract independent reference entropy
            S0cluster,J0cluster = self.Sindependent(cluster,coocMat)
            deltaScluster -= S0cluster
            deltaJcluster -= J0cluster
        elif meanFieldRef:
            # subtract mean field reference entropy
            S0cluster,J0cluster = SmeanField(cluster,coocMat,
                meanFieldPriorLmbda,numSamples)
            deltaScluster -= S0cluster
            deltaJcluster -= J0cluster
        
        # subtract entropies of sub-clusters
        for size in range(len(cluster)-1,0,-1):
          subclusters = self.subsets(cluster,size)
          for subcluster in subclusters:
            deltaSsubcluster,deltaJsubcluster = \
                self.deltaS(subcluster,coocMat,deltaSdict,deltaJdict,
                       verbose=verbose,
                       meanFieldRef=meanFieldRef,priorLmbda=priorLmbda,
                       numSamples=numSamples,
                       independentRef=independentRef,
                       meanFieldPriorLmbda=meanFieldPriorLmbda)
            deltaScluster -= deltaSsubcluster
            deltaJcluster -= deltaJsubcluster

        deltaSdict[cID] = deltaScluster
        deltaJdict[cID] = deltaJcluster

        return deltaScluster, deltaJcluster

    def clusterID(self, cluster):
        return tuple(np.sort(cluster))

    def subsets(self, set, size, sort=False):
        """
        Given a list, returns a list of all unique subsets of that list with given size.

        Parameters
        ----------
        set : list
        size : int
        sort : bool,False

        Returns
        -------
        sub : list
            All subsets of given size.
        """

        if len(set) != len(np.unique(set)): raise Exception
        
        if size == len(set): return [set]
        if size > len(set): return []
        if size <= 0: return []
        if size == 1: return [ [s,] for s in set ]
        
        sub = []
        rest = copy.copy(set)
        s = rest[0]
        rest.remove(s)
        
        subrest1 = self.subsets(rest,size)
        sub.extend(subrest1)
        
        subrest2 = self.subsets(rest,size-1)
        [ srest.append(s) for srest in subrest2 ]
        sub.extend(subrest2)
        
        if sort:
            return np.sort(sub)
        return sub

    # "Algorithm 2"
    # was "adaptiveClusterExpansion"
    def solve(self, X, threshold, 
              cluster=None,
              deltaSdict=None,
              deltaJdict=None,
              verbose=True,
              priorLmbda=0.,
              numSamples=None,
              meanFieldRef=False,
              independentRef=True,
              veryVerbose=False,
              meanFieldPriorLmbda=None,
              return_all=False):
        """
        Parameters
        ----------
        X : array-like
            Data set (n_samples,n_dim).
        threshold : float
        meanFieldRef : bool,False
            Expand about mean-field reference
        independentRef : bool,True
            Expand about independent reference
        priorLmbda : float,0.
            Strength of non-interacting prior
        meanFieldPriorLmbda : float,None
            Strength of non-interacting prior in mean field calculation
            (defaults to priorLmbda)
        
        Returns
        -------
        With return_all=False, returns
            J           : Estimated interaction matrix
        
        With return_all=True, returns
            ent         : Estimated entropy
            J           : Estimated interaction matrix
            clusters    : List of clusters
            deltaSdict  : 
            deltaJdict  :
        """

        # convert input to coocMat
        coocMat = mean_field_ising.cooccurrence_matrix((X+1)/2)
        
        if deltaSdict is None: deltaSdict = {}
        if deltaJdict is None: deltaJdict = {}
        
        if independentRef and meanFieldRef: raise Exception
        
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        
        N = len(coocMat)
        T = threshold
        if cluster is None: cluster = list(range(N))

        clusters = {} # LIST
        size = 1
        clusters[1] = [ [i] for i in cluster ]

        while len(clusters[size]) > 0:
            clusters[ size+1 ] = []
            numClusters = len(clusters[size])
            if verbose:
                print("adaptiveClusterExpansion: Clusters of size", size+1)
            for i in range(numClusters):
                for j in range(i+1,numClusters): # some are not unique!
                    gamma1 = clusters[size][i]
                    gamma2 = clusters[size][j]
                    gammaI = np.intersect1d(gamma1,gamma2)
                    gammaU = np.sort( np.union1d(gamma1,gamma2) )
                    gammaU = list(gammaU)
                    if (len(gammaI) == size-1):
                        deltaSgammaU, deltaJgammaU = self.deltaS(gammaU, coocMat, deltaSdict, deltaJdict,
                                                                 verbose=veryVerbose,
                                                                 meanFieldRef=meanFieldRef,
                                                                 priorLmbda=priorLmbda,
                                                                 numSamples=numSamples,
                                                                 independentRef=independentRef,
                                                                 meanFieldPriorLmbda=meanFieldPriorLmbda)
                        if (abs(deltaSgammaU) > T) and (gammaU not in clusters[size+1]):
                            clusters[ size+1 ].append(gammaU)
            size += 1
        
        if independentRef:
            ent,J0 = self.Sindependent(cluster,coocMat)
        elif meanFieldRef:
            ent,J0 = SmeanField(cluster,coocMat,
                                meanFieldPriorLmbda,numSamples)
        else:
            ent = 0.
            J0 = np.zeros((N,N))
        J = J0.copy()

        for size in list(clusters.keys()):
            for cluster in clusters[size]:
                cID = self.clusterID(cluster)
                ent += deltaSdict[cID]
                J += deltaJdict[cID]

        # convert J to {-1,1} basis
        h = -J.diagonal()
        J = -mean_field_ising.zeroDiag(J)
        self.multipliers = convert_params( h, squareform(J)*2, '11', concat=True )

        if return_all:
            return ent, self.multipliers, clusters, deltaSdict, deltaJdict
        else:
            return self.multipliers
# end ClusterExpansion


class RegularizedMeanField(Solver):
    """
    Implementation of regularized mean field method for solving the inverse Ising problem, as
    described in Daniels, Bryan C., David C. Krakauer, and Jessica C. Flack.  ``Control of
    Finite Critical Behaviour in a Small-Scale Social System.'' Nature Communications 8 (2017):
    14301.  doi:10.1038/ncomms14301
    
    Specific to pairwise Ising constraints.
    """
    def __init__(self, *args, **kwargs):
        """
        See Solver. Default sample_method is '_metropolis'.
        """
        super(RegularizedMeanField,self).__init__(*args,**kwargs)
        # some case handling to ensure that RMF gets control over the random number generator
        self.setup_sampler(kwargs.get('sample_method','metropolis'))

    def solve(self, samples,
              sample_size=100000,
              seed=0,
              change_seed=False,
              min_size=0,
              min_covariance=False,
              min_independent=True,
              cooc_cov=None,
              priorLmbda=0.,
              bracket=None,
              n_grid_points=200):
        """
        Varies the strength of regularization on the mean field J to best fit given cooccurrence data.
        
        n_grid_points : int, 200
            If bracket is given, first test at n_grid_points points evenly spaced in the bracket interval,
            then give the lowest three points to scipy.optimize.minimize_scalar
        sample_size : int, 100_000
        seed : int, 0
            initial seed for rng, seed is incremented by mean_field_ising.seedGenerator if change Seed option
            is True
        change_seed : bool, False
        min_size : int, 0
            Use a modified model in which samples with fewer ones than min_size are not allowed.
        min_covariance : bool, False
            ** As of v1.0.3, not currently supported **
            Minimize covariance from emperical frequencies (see notes); trying to avoid biases, as inspired by
            footnote 12 in TkaSchBer06
        min_independent : bool, True
            ** As of v1.0.3, min_independent is the only mode currently supported **
            Each <xi> and <xi xj> residual is treated as independent
        cooc_cov : ndarray,None
            ** As of v1.0.3, not currently supported **
            Provide a covariance matrix for residuals.  Should typically be coocSampleCovariance(samples).
            Only used if min_covariance and min_independent are False.
        priorLmbda : float,0.
            ** As of v1.0.3, not currently implemented **
            Strength of noninteracting prior.
        """

        from scipy import transpose

        numDataSamples = len(samples)
        # convert input to coocMat
        coocMatData = mean_field_ising.cooccurrence_matrix((samples+1)/2)
        
        if cooc_cov is None:
            cooc_cov = mean_field_ising.coocSampleCovariance(samples)
        
        if change_seed: seedIter = mean_field_ising.seedGenerator(seed, 1)
        else: seedIter = mean_field_ising.seedGenerator(seed, 0)
        
        if priorLmbda != 0.:
            raise NotImplementedError("priorLmbda is not currently supported")
            lmbda = priorLmbda / numDataSamples

        # stuff defining the error model, taken from findJmatrixBruteForce_CoocMat
        # 3.1.2012 I'm pretty sure the "repeated" line below should have the transpose, but
        # coocJacobianDiagonal is not sensitive to this.  If you use non-diagonal jacobians in the
        # future and get bad behavior you may want to double-check this.
        if min_independent:
            coocStdevs = mean_field_ising.coocStdevsFlat(coocMatData,numDataSamples)
            coocStdevsRepeated = ( coocStdevs*np.ones((len(coocStdevs),len(coocStdevs))) ).T
        elif min_covariance:
            raise Exception("min_covariance is not currently supported")
            empiricalFreqs = np.diag(coocMatData)
            covTildeMean = covarianceTildeMatBayesianMean(coocMatData,numDataSamples)
            covTildeStdevs = covarianceTildeStdevsFlat(coocMatData,numDataSamples,
                empiricalFreqs)
            covTildeStdevsRepeated = (
                    covTildeStdevs*np.ones((len(covTildeStdevs),len(covTildeStdevs))) ).T
        else:
            raise NotImplementedError("correlated residuals calculation is not currently supported")
            if cooc_cov is None: raise Exception
            cov = cooc_cov # / numDataSamples (can't do this here due to numerical issues)
                          # instead include numDataSamples in the calculation of coocMatMeanZSq

        # for use in gammaPrime <-> priorLmbda
        freqsList = np.diag(coocMatData)
        pmean = np.mean(freqsList)
        
        # adapted from findJMatrixBruteForce_CoocMat
        def samples(J):
           seed = next(seedIter)
           if min_covariance:
               J = tildeJ2normalJ(J, empiricalFreqs)
           burninDefault = 100*self.n
           J = J + J.T
           self.multipliers = np.concatenate([J.diagonal(), squareform(mean_field_ising.zeroDiag(-J))])
           self.sampler.rng = np.random.RandomState(seed)
           self.generate_samples(1, burninDefault, sample_size=int(sample_size))
           isingSamples = self.samples.copy()
           return isingSamples

        # adapted from findJMatrixBruteForce_CoocMat
        def func(meanFieldGammaPrime):
            
            # translate gammaPrime prior strength to lambda prior strength
            meanFieldPriorLmbda = meanFieldGammaPrime / (pmean**2 * (1.-pmean)**2)
            
            # calculate regularized mean field J
            J = mean_field_ising.JmeanField(coocMatData,
                                            meanFieldPriorLmbda=meanFieldPriorLmbda,
                                            numSamples=numDataSamples)

            # sample from J
            isingSamples = samples(J)
            
            # calculate residuals, including prior if necessary
            if min_independent: # Default
                dc = mean_field_ising.isingDeltaCooc(isingSamples, coocMatData)/coocStdevs
            elif min_covariance:
                dc = isingDeltaCovTilde(isingSamples, covTildeMean, empiricalFreqs)/covTildeStdevs
            else:
                dc = mean_field_ising.isingDeltaCooc(isingSamples, coocMatMean)
                if priorLmbda != 0.:
                    freqs = np.diag(coocMatData)
                    factor = np.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                    factorFlat = aboveDiagFlat(factor)
                    priorTerm = lmbda * factorFlat * flatJ[ell:]**2
                
                dc = np.concatenate([dc,priorTerm])
                
            if self.verbose:
                print("RegularizedMeanField.solve: Tried "+str(meanFieldGammaPrime))
                print("RegularizedMeanField.solve: sum(dc**2) = "+str(np.sum(dc**2)))
                
            return np.sum(dc**2)

        if bracket is not None:
            gridPoints = np.linspace(bracket[0], bracket[1], n_grid_points)
            gridResults = [ func(p) for p in gridPoints ]
            gridBracket = self.bracket1d(gridPoints, gridResults)
            solution = minimize_scalar(func, bracket=gridBracket)
        else:
            solution = minimize_scalar(func)

        gammaPrimeMin = solution['x']
        meanFieldPriorLmbdaMin = gammaPrimeMin / (pmean**2 * (1.-pmean)**2)
        J = mean_field_ising.JmeanField(coocMatData,
                                        meanFieldPriorLmbda=meanFieldPriorLmbdaMin,
                                        numSamples=numDataSamples)
        J = J + J.T

        # convert J to {-1,1} basis
        h = -J.diagonal()
        J = -mean_field_ising.zeroDiag(J)
        self.multipliers = convert_params( h, squareform(J)*2, '11', concat=True )

        return self.multipliers

    def bracket1d(self, xList, funcList):
        """
        Assumes xList is monotonically increasing
        
        Get bracketed interval (a,b,c) with a < b < c, and f(b) < f(a) and f(c).
        (Choose b and c to make f(b) and f(c) as small as possible.)
        
        If minimum is at one end, raise error.
        """

        gridMinIndex = np.argmin(funcList)
        gridMin = xList[gridMinIndex]
        if (gridMinIndex == 0) or (gridMinIndex == len(xList)-1):
            raise Exception("Minimum at boundary")
        gridBracket1 = xList[ np.argmin(funcList[:gridMinIndex]) ]
        gridBracket2 = xList[ gridMinIndex + 1 + np.argmin(funcList[gridMinIndex+1:]) ]
        gridBracket = (gridBracket1,gridMin,gridBracket2)
        return gridBracket
#end RegularizedMeanField
