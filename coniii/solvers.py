# =============================================================================================== #
# ConIII module for algorithms for solving the inverse Ising problem.
# Authors: Edward Lee (edlee@alumni.princeton.edu) and Bryan Daniels (bryan.daniels.1@asu.edu)
#
# MIT License
# 
# Copyright (c) 2020 Edward D. Lee, Bryan C. Daniels
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
from scipy.optimize import minimize, fmin_ncg, minimize_scalar, root
import multiprocess as mp
import copy
from . import mean_field_ising
from warnings import warn
from scipy.optimize import check_grad
from .utils import *
from .samplers import *
from .models import Ising


class Solver():
    """Base class for declaring common methods and attributes for inverse maxent
    algorithms.
    """
    def basic_setup(self, sample, model=None, calc_observables=None, model_kwargs={}):
        """
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        model_kwargs : dict, {}
            Additional arguments that will be passed to Ising class. These only matter if
            model is None. Important ones include "n_cpus" and "rng".
        """
        
        if not set(np.unique(sample).tolist())<=set((-1,1)):
            warn("Data is not only -1, 1 entries.")
        self.sample = sample
        self.n = sample.shape[1]

        if model is None:
            self.model = Ising(np.zeros((self.n**2+self.n)//2), **model_kwargs)
            if self.model.calc_observables is None:
                msg = ("Python file enumerating the Ising equations for system of size %d must be written to"+
                       " use this solver.")
                raise Exception(msg%self.n)
        else:
            self.model = model

        if calc_observables is None:
            warn("Assuming that calc_observables should be for Ising model.")
            self.calc_observables = define_ising_helper_functions()[1]
        else:
            self.calc_observables = calc_observables

        self.constraints = self.calc_observables(sample).mean(0)
        if np.isclose(np.abs(self.constraints), 1, atol=1e-3).any():
            warn("Some pairwise correlations have magnitude close to one. Potential for poor solutions.")

    def solve(self):
        return
#end Solver


class Enumerate(Solver):
    """Class for solving +/-1 symmetric Ising model maxent problems by gradient descent
    with flexibility to put in arbitrary constraints.
    """
    def __init__(self, sample, model=None, calc_observables=None, **default_model_kwargs):
        """
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        **default_model_kwargs
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs) 

    def solve(self,
              initial_guess=None,
              constraints=None,
              max_param_value=50,
              full_output=False,
              use_root=True,
              scipy_solver_kwargs={'method':'krylov',
                                   'options':{'fatol':1e-13,'xatol':1e-13}}):
        """Must specify either constraints (the correlations) or samples from which the
        correlations will be calculated using self.calc_observables. This routine by
        default uses scipy.optimize.root to find the solution. This is MUCH faster than
        the scipy.optimize.minimize routine which can be used instead.
        
        If still too slow, try adjusting the accuracy.
        
        If not converging, try increasing the max number of iterations.

        If receiving Jacobian error (or some other numerical estimation error), parameter
        values may be too large for faithful evaluation. Try decreasing max_param_value.

        Parameters
        ----------
        initial_guess : ndarray, None
            Initial starting guess for parameters. By default, this will start with all
            zeros if left unspecified.
        constraints : ndarray, None
            For debugging!
            Can specify constraints directly instead of using the ones calculated from the
            sample. This can be useful when the pairwise correlations are known exactly.
        max_param_value : float, 50
            Absolute value of max parameter value. Bounds can also be set in the kwargs
            passed to the minimizer, in which case this should be set to None.
        full_output : bool, False
            If True, return output from scipy.optimize.minimize.
        use_root : bool, True
            If False, use scipy.optimize.minimize instead. This is typically much slower.
        scipy_solver_kwargs : dict, {'method':'krylov', 'options':{'fatol':1e-13,'xatol':1e-13}}
            High accuracy is slower. Although default accuracy may not be so good,
            lowering these custom presets will speed things up. Choice of the root finding
            method can also change runtime and whether a solution is found or not.
            Recommend playing around with different solvers and tolerances or getting a
            close approximation using a different method if solution is hard to find.

        Returns
        -------
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        dict, optional
            Output from scipy.optimize.root.
        """
        
        if not initial_guess is None:
            assert initial_guess.size==self.constraints.size
        else: initial_guess = np.zeros((len(self.constraints)))
        if constraints is None:
            constraints = self.constraints
        
        # default solver routine
        if use_root:
            if not max_param_value is None:
                def f(params):
                    if np.any(np.abs(params)>max_param_value):
                        return np.zeros_like(constraints) + 1e30
                    return self.model.calc_observables(params)-constraints
            else:
                def f(params):
                    return self.model.calc_observables(params)-constraints
            
            soln = root(f, initial_guess, **scipy_solver_kwargs)
        else:
            if not max_param_value is None:
                def f(params):
                    if np.any(np.abs(params)>max_param_value):
                        return 1e30
                    return np.linalg.norm( self.model.calc_observables(params)-constraints )
            else:
                def f(params):
                    return np.linalg.norm( self.model.calc_observables(params)-constraints )
            
            soln = minimize(f, initial_guess, **scipy_solver_kwargs)

        self.multipliers = soln['x']
        if full_output:
            return soln['x'], soln
        return soln['x']
#end Enumerate


def unwrap_self_worker_obj(arg, **kwarg):
    return MPF.worker_objective_task(*arg, **kwarg)

class MPF(Solver):
    def __init__(self, sample,
                 model=None,
                 calc_observables=None,
                 calc_de=None,
                 adj=None,
                 **default_model_kwargs):
        """Parallelized implementation of Minimum Probability Flow algorithm.

        Most time consuming step is the computation of the energy of a given state. Make
        this as fast as possible.

        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        calc_de : function, None
            Function for calculating derivative of energy wrt parameters. Takes in 2d
            state array and index of the parameter.
        adj : function, None
            Function for getting all the neighbors of any given state. Note that the
            backed in self.solvers runs everything in the {0,1} basis for spins, so this
            needs to find neighboring states in the {0,1} basis.
        **default_model_kwargs
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs)
        if adj is None:
            from .utils import adj
            self.adj = adj
        if calc_de is None:
            self.calc_de = calc_de  # imported from utils.py
        
    @staticmethod
    def worker_objective_task( s, Xcount, adjacentStates, params, calc_e ):
        return Xcount * np.sum(np.exp( .5*(calc_e(s[None,:],params) 
                                           - calc_e(adjacentStates,params) ) ))
 
    def K( self, Xuniq, Xcount, adjacentStates, params ):
        """Compute objective function.
        
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
            dobj = Xcount[i] * np.exp( .5*(self.model.calc_e(s[None,:], params) 
                                           - self.model.calc_e(adjacentStates[i], params) ) )
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
        """Translation from Sohl-Dickstein's code K_dk_ising.m. This is here for testing
        purposes only.  Caution: This uses a different convention for negatives and 1/2
        factors. To use this properly, all parameters will have an extra negative, the
        returned J's will be halved and the energy calculation should include a 1/2 factor
        in front of h's.
        """

        nbatch, ndims = X.shape
        X = X.T
        
        h = J[:ndims]
        J = squareform( J[ndims:] )
        J[diag_indices(ndims)] = h
        
        Y = dot(J,X)
        diagJ = J.diagonal()
        # XnotX contains (X - [bit flipped X])
        XnotX = 2.*X-1;
        # Kfull is a [ndims, nbatch] matrix containing the contribution to the 
        # objective function from flipping each bit in the rows, for each datapoint 
        # on the columns
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

        obj = 0.
        objGrad = np.zeros((params.size))
        power = np.zeros((len(Xuniq), len(adjacentStates[0])))  # energy differences
        for i,s in enumerate(Xuniq):
            power[i,:] = .5*( self.model.calc_e(s[None,:], params) -
                              self.model.calc_e(adjacentStates[i], params) )
            
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
        """Use self.adj to evaluate all adjacent states in Xuniq.

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
            adjacentStates.append( self.adj(s).astype(int) )
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
              initial_guess=None,
              method='L-BFGS-B',
              full_output=False,
              all_connected=True,
              parameter_limits=100,
              solver_kwargs={'maxiter':100,'disp':False,'ftol':1e-15},
              uselog=True):
        """Minimize MPF objective function using scipy.optimize.minimize.

        Parameters
        ----------
        initial_guess : ndarray, None
        method : str, 'L-BFGS-B'
            Option for scipy.optimize.minimize.
        full_output : bool, False
        all_connected : bool, True
            Switch for summing over all states that data sets could be connected to or
            just summing over non-data states (second summation in Eq 10 in Sohl-Dickstein
            2011).
        parameter_limits : float, 100
            Maximum allowed magnitude of any single parameter.
        solver_kwargs : dict, {'maxiter':100,'disp':False,'ftol':1e-15}
            For scipy.optimize.minimize.
        uselog : bool, True
            If True, calculate log of the objective function. This can help with numerical
            precision errors.

        Returns
        -------
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        dict (optional)
            Output from scipy.optimize.minimize returned if full_output is True.
        """
        
        assert parameter_limits>0
        # Convert from {+/-1} to {0,1} axis.
        X = (self.sample+1)//2

        if not self.calc_de is None:
            includeGrad = True
        else:
            includeGrad = False
        if initial_guess is None:
            initial_guess = self.calc_observables(X).mean(0)
        else:
            initial_guess = ising_convert_params( split_concat_params(initial_guess, self.n), '01', True)
         
        # Get list of unique data states and how frequently they appear.
        Xuniq, ix, Xcount = np.unique(X, axis=0, return_inverse=True, return_counts=True)
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
        self.multipliers = ising_convert_params( split_concat_params(soln['x'], self.n), '11', True)

        if full_output:
            return self.multipliers, soln
        return ising_convert_params( split_concat_params(soln['x'], self.n), '11', True)
#end MPF


class MCH(Solver):
    """Class for solving maxent problems using the Monte Carlo Histogram method.

    Broderick, T., Dudik, M., Tkacik, G., Schapire, R. E. & Bialek, W. Faster solutions of the
    inverse pairwise Ising problem. arXiv 1-8 (2007).
    """
    def __init__(self, sample, 
                 model=None,
                 calc_observables=None,
                 sample_size=1000,
                 sample_method='metropolis',
                 mch_approximation=None,
                 **default_model_kwargs):
        """
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        sample_size : int, 1000
            Number of samples to use MCH sampling step.
        sample_method : str, 'metropolis'
            Only 'metropolis' allowed currently.
        mch_approximation : function, None
            For performing the MCH approximation step. Is specific to the maxent model.
        rng : np.random.RandomState, None
            Random number generator.
        n_cpus : int, None
            If 1 or less no parallel processing, other numbers above 0 specify number of
            cores to use.
        **default_model_kwargs
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        assert sample_size>0
        if sample_size<1000: warn("Small sample size will lead to poor convergence.")
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs)

        # Sampling parameters.
        self.sampleSize = sample_size
        self.mch_approximation = mch_approximation or define_ising_helper_functions()[-1]
        
        self.model.setup_sampler(sample_size=sample_size)
    
    def solve(self,
              initial_guess=None,
              constraints=None,
              tol=None,
              tolNorm=None,
              n_iters=30,
              burn_in=30,
              maxiter=10,
              custom_convergence_f=None,
              iprint=False,
              full_output=False,
              learn_params_kwargs={'maxdlamda':1, 'eta':1},
              generate_kwargs={}):
        """Solve for maxent model parameters using MCH routine.
        
        Parameters
        ----------
        initial_guess : ndarray, None
            Initial starting point.
        constraints : ndarray, None
            For debugging!
            Vector of correlations to fit.
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
            Function for determining convergence criterion. At each iteration, this
            function should return the next set of learn_params_kwargs and optionally the
            sample size.

            As an example:
	    def learn_settings(i):
		'''
                Take in the iteration counter and set the maximum change allowed in any
                given parameter (maxdlamda) and the multiplicative factor eta, where 
		d(parameter) = (error in observable) * eta.
		
                Additional option is to also return the sample size for that step by
                returning a tuple. Larger sample sizes are necessary for higher accuracy.
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
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        int
            Error flag.
            0, converged within given criterion
            1, max iterations reached
        ndarray
            Log of errors in matching constraints at each step of iteration.
        """

        if (self.n*10)>burn_in:
            warn("Number of burn in MCMC iterations between samples may be too small for "+
                 "convergence to stationary distribution.")
        if (self.n*10)>n_iters:
            warn("Number of MCMC iterations between samples may be too small for convergence to "+
                 "stationary distribution.")
        if constraints is None:
            constraints = self.constraints

        errors = []  # history of errors to track

        # Set initial guess for parameters. self._multipliers is where the current guess for the
        # parameters is stored.
        if not (initial_guess is None):
            assert len(initial_guess)==len(constraints)
            self._multipliers = initial_guess.copy()
        else:
            self._multipliers = np.zeros((len(constraints)))
        tol = tol or 1/np.sqrt(self.model.sampleSize)
        tolNorm = tolNorm or np.sqrt( 1/self.model.sampleSize )*len(self._multipliers)
        
        # Redefine function for automatically adjusting learn_params_kwargs so that it returns the
        # MCH iterator settings and the sample size if it doesn't already.
        if custom_convergence_f is None:
            custom_convergence_f = lambda i:learn_params_kwargs,self.model.sampleSize
        if type(custom_convergence_f(0)) is dict:
            custom_convergence_f_ = custom_convergence_f
            custom_convergence_f = lambda i:(custom_convergence_f_(i),self.model.sampleSize)
        assert 'maxdlamda' and 'eta' in list(custom_convergence_f(0)[0].keys())
        assert type(custom_convergence_f(0)[1]) is int
        
        
        # Generate initial set of samples.
        self.model.generate_samples( n_iters,burn_in,
                                     multipliers=self._multipliers,
                                     generate_kwargs=generate_kwargs )
        thisConstraints = self.calc_observables(self.model.sample).mean(0)
        errors.append( thisConstraints - constraints )
        if iprint=='detailed': print(self._multipliers)


        # MCH iterations.
        counter = 0  # number of MCMC and MCH steps
        keepLooping = True  # loop control
        learn_params_kwargs, self.model.sampleSize = custom_convergence_f(counter)
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
            self.model.generate_samples( n_iters, burn_in,
                                         multipliers=self._multipliers,
                                         generate_kwargs=generate_kwargs )
            thisConstraints = self.calc_observables(self.model.sample).mean(0)
            counter += 1
            
            errors.append( thisConstraints - constraints )
            if iprint=='detailed':
                print("Error is %1.4f"%np.linalg.norm(errors[-1]))
            # Exit criteria.
            if ( np.linalg.norm(errors[-1])<tolNorm
                 and np.all(np.abs(thisConstraints - constraints)<tol) ):
                if iprint: print("Solved.")
                errflag = 0
                keepLooping=False
            elif counter>maxiter:
                if iprint: print("Over maxiter")
                errflag = 1
                keepLooping=False
            else:
                learn_params_kwargs, self.model.sampleSize = custom_convergence_f(counter)
        
        self.multipliers = self._multipliers.copy()
        if full_output:
            return self.multipliers, errflag, np.vstack((errors))
        return self.multipliers

    def estimate_jac(self, eps=1e-3):
        """Approximation Jacobian using the MCH approximation. 

        Parameters
        ----------
        eps : float, 1e-3

        Returns
        -------
        jac : ndarray
            Jacobian is an n x n matrix where each row corresponds to the behavior of fvec
            wrt to a single parameter.
        """

        dlamda = np.zeros(self._multipliers.shape)
        jac = np.zeros((self._multipliers.size,self._multipliers.size))
        print("evaluating jac")
        for i in range(len(self._multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.sample, dlamda)

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.sample, dlamda)     

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
            estConstraints = self.mch_approximation( self.sample, dlamda )
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
MonteCarloHistogram = MCH  # alias


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
        """Not ready for release.
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
        """Solve for parameters using MCH routine.
        
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
    Pseudolikelihood approximation to solving the inverse Ising problem as described in
    Aurell and Ekeberg, PRL 108, 090201 (2012).
    """
    def __init__(self, sample,
                 model=None,
                 calc_observables=None,
                 get_multipliers_r=None,
                 calc_observables_r=None,
                 k=2,
                 **default_model_kwargs):
        """For this technique, must specify how to calculate the energy specific to the
        conditional probability of spin r given the rest of the spins. These will be
        passed in with "get_observables_r" and "calc_observables_r".
        
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        get_multipliers_r : function, None
            Takes index r and multipliers.
            Defn: lambda r,multipliers : r_multipliers
        calc_observables_r : function, None
            Takes index r and samples X.
            Defn: lambda r,X : r_observable
        k : int
            Number of possible states for each spin. This should only be changed for the Potts model.
        **default_model_kwargs
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs)
        if calc_observables_r is None or get_multipliers_r is None:
            self.get_multipliers_r, self.calc_observables_r = define_pseudo_ising_helper_functions(self.n)
        else:
            assert sample.max()<=(k-1)
            self.k = k
            self.get_multipliers_r, self.calc_observables_r = get_multipliers_r, calc_observables_r

    def solve(self, force_general=False, **kwargs):
        """Uses a general all-purpose optimization to solve the problem using functions
        defined in self.get_multipliers_r and self.calc_observables_r.

        Parameters
        ----------
        force_general : bool, False
            If True, force use of "general" algorithm.
        initial_guess : ndarray, None
            Initial guess for the parameter values.
        solver_kwargs : dict, {}
            kwargs for scipy.minimize().

        Returns
        -------
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        """

        from .models import Potts3, Ising
        
        if type(self.model) is Ising and not force_general:
            return self._solve_ising(**kwargs)
        elif type(self.model) is Potts3:
            return self._solve_potts(**kwargs)
        return self._solve_general(**kwargs)
        
    def _solve_ising(self,
                     initial_guess=None,
                     full_output=False,
                     solver_kwargs={}):
        """Solve for Langrangian parameters according to pseudolikelihood algorithm.

        Parameters
        ----------
        initial_guess : ndarray, None
            Initial guess for the parameter values.
        full_output : bool, False
            If True, return output from scipy.minimize() routine.
        solver_kwargs : dict, {}
            Keyword arguments for scipy.optimize.minimize.

        Returns
        -------
        ndarray
            Solved multipliers.
        dict (optional)
            Output from scipy.optimize.minimize.
        """

        if initial_guess is None:
            initial_guess = np.zeros(self.calc_observables(self.sample[0][None,:]).size)
            
        # reformat initial_guess for easy looping later
        initial_guessAsMat = replace_diag(squareform(initial_guess[self.n:]), initial_guess[:self.n])
        for i in range(1,self.n):
            tmp = initial_guessAsMat[i,i]
            initial_guessAsMat[i,i] = initial_guessAsMat[i,0]
            initial_guessAsMat[i,0] = tmp
        # initialize variables that will be used later 
        obs = [self.calc_observables_r(r, self.sample) for r in range(self.n)]
        soln = []  # list of output from scipy.optimize.minimize for each spin
        Jmat = np.zeros((self.n, self.n))  # couplings stored in matrix format with fields along diagonal
        
        # iterate through each spin and solve for parameters for each one
        for r in range(self.n):
            # to use below...
            multipliersrix = self.get_multipliers_r(r, initial_guess)[1]
            guess = initial_guess.copy()

            # params only change the terms relevant to the particular spin being considered
            def f(params):
                guess[multipliersrix] = params
                multipliers = self.get_multipliers_r(r, guess)[0]
                E = -obs[r].dot(multipliers)
                loglikelihood = -np.log( 1+np.exp(2*E) ).sum()
                dloglikelihood = ( -(1/(1+np.exp(2*E)) * np.exp(2*E))[:,None] * 2*obs[r] ).sum(0)
                return -loglikelihood, dloglikelihood
        
            soln.append(minimize(f, initial_guessAsMat[r], jac=True, **solver_kwargs))
            thisMultipliers = soln[-1]['x']
            Jmat[r,r] = thisMultipliers[0]
            Jmat[r,np.delete(np.arange(self.n),r)] = thisMultipliers[1:]
        
        # symmetrize couplings
        Jmat = (Jmat + Jmat.T)/2
        self.multipliers = np.concatenate((Jmat.diagonal(), squareform(zero_diag(Jmat))))
            
        if full_output:
            return self.multipliers, soln
        return self.multipliers
        
    def _solve_general(self,
                       initial_guess=None,
                       full_output=False,
                       solver_kwargs={}):
        """Solve for Langrangian parameters according to a variation on the
        pseudolikelihood algorithm detailed in Aurell and Ekeberg (PRL, 2012). There, the
        conditional log-likelihoods per spin are minimized independently and then the
        resulting couplings are combined in a way that ensures that the interactions are
        symmetric. The generalization is straightforward for higher-order interactions
        (normalize by the order of the interaction), but here present a different approach
        that is somewhat computationally simpler.
        
        The *sum* of the conditional likelihoods over each spin is minimized, ensuring
        that the parameters are equal across all conditional likelihood equations by
        construction. In general, this gives different results from the original
        pseudolikelihood formulation, but they agree closely in many cases.

        Parameters
        ----------
        initial_guess : ndarray, None
            Initial guess for the parameter values.
        full_output : bool, False
            If True, return output from scipy.minimize() routine.
        solver_kwargs : dict, {}
            Keyword arguments for scipy.optimize.minimize.

        Returns
        -------
        ndarray
            Solved multipliers.
        dict (optional)
            Output from scipy.optimize.minimize.
        """

        if initial_guess is None:
            initial_guess = np.zeros(self.calc_observables(self.sample[0][None,:]).size)
        obs = [self.calc_observables_r(r, self.sample) for r in range(self.n)]
        
        def f(params):
            # running sums of function evaluations over all spins
            loglikelihood = 0
            dloglikelihood = np.zeros_like(initial_guess)  # gradient

            # iterate through each spin
            for r in range(self.n):
                multipliers, multipliersrix = self.get_multipliers_r(r, params)
                E = -obs[r].dot(multipliers)
                loglikelihood += -np.log( 1+np.exp(2*E) ).sum() 
                dloglikelihood[multipliersrix] += ( -(1/(1+np.exp(2*E)) *
                                                    np.exp(2*E))[:,None] *
                                                    2*obs[r] ).sum(0)
            return -loglikelihood, dloglikelihood
        
        soln = minimize(f, initial_guess, jac=True, **solver_kwargs)
        self.multipliers = soln['x']
        if full_output:
            return soln['x'], soln
        return soln['x']

    def _solve_potts(self,
                     initial_guess=None,
                     full_output=False,
                     solver_kwargs={},
                     cost_fcn=None,
                     cost_fcn_jac=None):
        """Solve Potts model formulation with k-states and non-zero coupling if spins are
        in the same state.

        Parameters
        ----------
        initial_guess : ndarray, None
            Initial guess for the parameter values.
        full_output : bool, False
            If True, return output from scipy.minimize() routine.
        solver_kwargs : dict, {}
            Keyword arguments for scipy.optimize.minimize.
        cost_fcn : lambda function, None
            Takes the given set of parameters and returns a cost that is added to the neg
            log likelihood.  Must be specified along with the jacobian.
        cost_fcn_jac : lambda function, None
            Jacobian for above cost function. 

        Returns
        -------
        ndarray
            Solved multipliers.
        dict (optional)
            Output from scipy.optimize.minimize.
        """

        if initial_guess is None:
            initial_guess = np.zeros(self.calc_observables(self.sample[0][None,:]).size)
        obs = []
        otherobs = []
        otherobsstate = []
        for r in range(self.n):
            out = self.calc_observables_r(r, self.sample)
            obs.append(out[0])
            otherobs.append(out[1])
            otherobsstate.append(out[2])
        if cost_fcn is None:
            cost_fcn = lambda x: 0
            cost_fcn_jac = lambda x: 0
        elif cost_fcn:
            assert cost_fcn_jac, "Must specify jacobian for cost function as well."
            jacErr = check_grad(cost_fcn, cost_fcn_jac, initial_guess)
            if jacErr>1e-4:
                warn("Jacobian fcn is bad. Norm error of %E."%jacErr)

        def f(params):
            # running sums of function evaluations over all spins
            loglikelihood = 0
            dloglikelihood = np.zeros_like(initial_guess)  # gradient

            # iterate through each spin
            for r in range(self.n):
                multipliers, multipliersrix = self.get_multipliers_r(r, params)
                
                # first, calculate the log likelihood
                E = -obs[r].dot(multipliers)
                Eother = np.vstack([-o.dot(multipliers) for o in otherobs[r]]).T

                Edelta = Eother - E[:,None]
                Edelta = np.hstack((Edelta, np.zeros((self.sample.shape[0],1))))  # add constant term
                loglikelihoodPerSample = -logsumexp(-Edelta, axis=1)

                loglikelihood += loglikelihoodPerSample.sum()

                # calculate log likelihood gradient (take a derivative wrt to each parameter)
                den = -np.exp(-loglikelihoodPerSample)

                # iterate over each field
                # note that we are taking the derivative of the loglikelihood divided by
                # the term in the numerator, which makes the eqn simpler to handle but
                # makes keeping track of negatives and zeros a pain, which is mainly what
                # is happening below
                for hix in range(self.k):
                    num = np.zeros(self.sample.shape[0])

                    # iterate over each exponential term that consists of the (k-1) possible other values of
                    # this spin 
                    for ix in range(self.k-1):
                        sgn = np.ones(self.sample.shape[0])
                        currentStateAndSameField = self.sample[:,r]==hix
                        sgn[currentStateAndSameField] = -1
                        notCurrentStateAndNoField = (~currentStateAndSameField) & (otherobsstate[r][:,ix]!=hix)
                        sgn[notCurrentStateAndNoField] = 0

                        num += sgn * np.exp(-Edelta[:,ix])
                    dloglikelihood[multipliersrix[hix]] += (num/den).sum()
            
                # derivative wrt to each coupling
                for i,jix in enumerate(np.delete(range(self.n), r)):
                    num = np.zeros(self.sample.shape[0])

                    # iterate over each exponential term
                    for ix in range(self.k-1):
                        sgn = np.zeros(self.sample.shape[0])
                        
                        neighborjIsSameStateAsCounterfactual = otherobsstate[r][:,ix]==self.sample[:,jix]
                        sgn[neighborjIsSameStateAsCounterfactual] = 1

                        sameStateAsR = self.sample[:,r]==self.sample[:,jix]
                        sgn[sameStateAsR] = -1

                        num += sgn * np.exp(-Edelta[:,ix])
                    dloglikelihood[multipliersrix[i+self.k]] += (num/den).sum()
            return -loglikelihood + cost_fcn(params), -dloglikelihood + cost_fcn_jac(params)
        
        #from scipy.optimize import check_grad, approx_fprime
        #if check_grad(lambda x: f(x)[0], lambda x: f(x)[1], initial_guess)>1e-6:
        #    print("num:",approx_fprime(initial_guess, lambda x: f(x)[0], 1e-7)[9:])
        #    print("analytic:",f(initial_guess)[1][9:])
        #    print(approx_fprime(initial_guess, lambda x: f(x)[0], 1e-7) - f(initial_guess)[1])
        #    raise Exception
        
        soln = minimize(f, initial_guess, jac=True, **solver_kwargs)
        self.multipliers = soln['x']
        if full_output:
            return soln['x'], soln
        return soln['x']

    def _solve_ising_deprecated(self, initial_guess=None, full_output=False):
        """Deprecated.

        Parameters
        ----------
        initial_guess : ndarray, None
            Pseudo for Ising doesn't use a starting point. This is syntactic sugar.
        full_output : bool, False

        Returns
        -------
        ndarray
            Solved multipliers.
        """
        
        X = self.sample
        X = (X + 1)/2  # change from {-1,1} to {0,1}
        
        # start at freq. model params?
        freqs = np.mean(X, axis=0)
        hList = -np.log(freqs / (1. - freqs))
        Jfinal = np.zeros((self.n,self.n))

        for r in range(self.n):
            Jr0 = np.zeros(self.n)
            Jr0[r] = hList[r]
            
            XRhat = X.copy()
            XRhat[:,r] = np.ones(len(X))
            # calculate once and pass to hessian algorithm for speed
            pairCoocRhat = self.pair_cooc_mat(XRhat)
            
            Lr = lambda Jr: - self.cond_log_likelihood(r, X, Jr)
            fprime = lambda Jr: self.cond_jac(r, X, Jr)
            fhess = lambda Jr: self.cond_hess(r, X, Jr, pairCoocRhat=pairCoocRhat)
            
            Jr = fmin_ncg(Lr, Jr0, fprime, fhess=fhess, disp=False)
            Jfinal[r] = Jr

        Jfinal = -0.5*( Jfinal + Jfinal.T )
        hfinal = Jfinal[np.diag_indices(self.n)]

        # Convert parameters into {-1,1} basis as is standard for this package.
        Jfinal[np.diag_indices(self.n)] = 0
        self.multipliers = convert_params( hfinal, squareform(Jfinal)*2, '11', concat=True )

        return self.multipliers

    def cond_log_likelihood(self, r, X, Jr):
        """Equals the conditional log likelihood -L_r.

        Deprecated.
        
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

        X, Jr = np.array(X), np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        invPs = 1. + np.exp( energies )
        logLs = np.log( invPs )

        return -logLs.sum()

    def cond_jac(self, r, X, Jr):
        """Returns d cond_log_likelihood / d Jr, with shape (dimension of system)

        Deprecated.
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
        """Returns d^2 cond_log_likelihood / d Jri d Jrj, with shape (dimension of
        system)x(dimension of system)

        Current implementation uses more memory for speed.  For large sample size, it may
        make sense to break up differently if too much memory is being used.

        Deprecated.

        Parameters
        ----------
        pairCooc : ndarray, None
            Pass pair_cooc_mat(X) to speed calculation.
        """

        X, Jr = np.array(X), np.array(Jr)
        
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

        Deprecated.
        """

        p = [ np.outer(f,f) for f in X ]
        return np.transpose(p,(1,0,2))

    def pseudo_log_likelihood(self, X, J):
        """TODO: Could probably be made more efficient.

        Deprecated.

        Parameters
        ----------
        X : ndarray
            binary matrix, (# of samples) x (dimension of system)
        J : ndarray
            (dimension of system) x (dimension of system)
            J should be symmetric
        """

        return np.sum([ cond_log_likelihood(r,X,J) for r in range(len(J)) ])
#end Pseudo


class ClusterExpansion(Solver):
    """Implementation of Adaptive Cluster Expansion for solving the inverse Ising problem,
    as described in John Barton and Simona Cocco, J. of Stat. Mech.  P03002 (2013).
    
    Specific to pairwise Ising constraints.
    """
    def __init__(self, sample,
                 model=None,
                 calc_observables=None,
                 sample_size=1000,
                 **default_model_kwargs):
        """
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        sample_size : int, 1000
            Number of MC samples.
        rng : np.random.RandomState, None
            Random number generator.
        n_cpus : int, None
            If 1 or less no parallel processing, other numbers above 0 specify number of
            cores to use.
        **default_model_kwargs
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs)
        if sample_size<1000:
            warn("Sample size may be too small for convergence.")
        self.sampleSize = sample_size
        self.model.setup_sampler(sample_size=sample_size)
    
    def S(self, cluster, coocMat,
          deltaJdict={}, 
          useAnalyticResults=False,
          priorLmbda=0.,
          numSamples=None):
        """Calculate pairwise entropy of cluster.  (First fits pairwise Ising model.)
        
        Parameters
        ----------
        cluster : list
            List of indices belonging to each cluster.
        coocMat : ndarray
            Pairwise correlations.
        deltaJdict : dict, {}
        useAnalyticResults : bool, False
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
        """Entropy approximation assuming that each cluster appears independently of the
        others.

        Parameters
        ----------
        cluster : list
        coocMat : ndarray
            Pairwise correlations.

        Returns
        -------
        float
            Sind, independent entropy.
        ndarray
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
        deltaSdict : dict, None
        deltaJdict : dict, None
        verbose : bool, True
        meanFieldRef : bool, False
        numSamples : int, None
        independentRef : bool, False
            If True, expand about independent entropy
        meanFieldRef : bool, False
            If True, expand about mean field entropy

        Returns
        -------
        float
            deltaScluster
        float
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
        deltaScluster, deltaJcluster = self.S(cluster,coocMat,
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

    def subsets(self, thisSet, size, sort=False):
        """Given a list, returns a list of all unique subsets of that list with given
        size.

        Parameters
        ----------
        thisSet : list
        size : int
        sort : bool, False

        Returns
        -------
        list
            All subsets of given size.
        """

        if len(thisSet) != len(np.unique(thisSet)): raise Exception
        
        if size == len(thisSet): return [thisSet]
        if size > len(thisSet): return []
        if size <= 0: return []
        if size == 1: return [ [s,] for s in thisSet ]
        
        sub = []
        rest = copy.copy(thisSet)
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
    def solve(self, threshold, 
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
              full_output=False):
        """
        Parameters
        ----------
        threshold : float
        meanFieldRef : bool, False
            Expand about mean-field reference.
        independentRef : bool, True
            Expand about independent reference.
        priorLmbda : float, 0.
            Strength of non-interacting prior.
        meanFieldPriorLmbda : float, None
            Strength of non-interacting prior in mean field calculation (defaults to
            priorLmbda).
        
        Returns
        -------
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        float (optional, only if full_output=True)
            Estimated entropy.
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        list (optional, only if full_output=True)
            List of clusters.
        dict (optional, only if full_output=True)
            deltaSdict
        dict (optional, only if full_output=True)
            deltaJdict
        """

        # convert input to coocMat
        coocMat = mean_field_ising.cooccurrence_matrix((self.sample+1)/2)
        
        if deltaSdict is None: deltaSdict = {}
        if deltaJdict is None: deltaJdict = {}
        
        if independentRef and meanFieldRef: raise Exception
        
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        
        N = len(coocMat)
        T = threshold
        if cluster is None: cluster = list(range(N))

        clusters = {}
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
            ent, J0 = self.Sindependent(cluster, coocMat)
        elif meanFieldRef:
            ent, J0 = SmeanField(cluster, coocMat, meanFieldPriorLmbda, numSamples)
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
        J = -zero_diag(J)
        self.multipliers = convert_params( h, squareform(J)*2, '11', concat=True )

        if full_output:
            return self.multipliers, ent, clusters, deltaSdict, deltaJdict
        else:
            return self.multipliers
# end ClusterExpansion


class RegularizedMeanField(Solver):
    """Implementation of regularized mean field method for solving the inverse Ising
    problem, as described in Daniels, Bryan C., David C. Krakauer, and Jessica C. Flack.
    ``Control of Finite Critical Behaviour in a Small-Scale Social System.'' Nature
    Communications 8 (2017): 14301.  doi:10.1038/ncomms14301
    
    Specific to pairwise Ising constraints.
    """
    def __init__(self, sample,
                 model=None,
                 calc_observables=None,
                 sample_size=1_000,
                 verbose=False,
                 **default_model_kwargs):
        """
        Parameters
        ----------
        sample : ndarray
            Of dimensions (samples, dimension).
        model : class like one from models.py, None
            By default, will be set to solve Ising model.
        calc_observables : function, None
            For calculating observables from a set of samples.
        sample : ndarray
        model : class from models.py, None
        calc_observables : function, None
        sample_size : int, 1_000
        verbose : bool, False
        rng : np.random.RandomState, None
            Random number generator.
        n_cpus : int, None
            If 1 or less no parallel processing, other numbers above 0 specify number of
            cores to use.
        **default_model_kwargs : kwargs for default model
            Additional arguments that will be passed to Ising class. These only matter if
            model is None.
        """
        
        assert sample_size>0
        if sample_size<1000: warn("Small sample size will lead to poor convergence.")
        
        self.basic_setup(sample, model, calc_observables, model_kwargs=default_model_kwargs)
        self.sampleSize = sample_size
        self.verbose = verbose

        self.model.setup_sampler(sample_size=sample_size)

    def solve(self,
              n_grid_points=200,
              min_size=0,
              reset_rng=True,
              min_covariance=False,
              min_independent=True,
              cooc_cov=None,
              priorLmbda=0.,
              bracket=None):
        """Varies the strength of regularization on the mean field J to best fit given
        cooccurrence data.
        
        Parameters
        ----------
        n_grid_points : int, 200
            If bracket is given, first test at n_grid_points points evenly spaced in the
            bracket interval, then give the lowest three points to
            scipy.optimize.minimize_scalar
        min_size : int, 0
            Use a modified model in which samples with fewer ones than min_size are not
            allowed.
        reset_rng: bool, True
            Reset random number generator seed before sampling to ensure that objective
            function does not depend on generator state.
        min_covariance : bool, False
            ** As of v1.0.3, not currently supported **
            Minimize covariance from emperical frequencies (see notes); trying to avoid
            biases, as inspired by footnote 12 in TkaSchBer06
        min_independent : bool, True
            ** As of v1.0.3, min_independent is the only mode currently supported **
            Each <xi> and <xi xj> residual is treated as independent
        cooc_cov : ndarray, None
            ** As of v1.0.3, not currently supported **
            Provide a covariance matrix for residuals.  Should typically be
            coocSampleCovariance(samples).  Only used if min_covariance and
            min_independent are False.
        priorLmbda : float,0.
            ** As of v1.0.3, not currently implemented **
            Strength of noninteracting prior.

        Returns
        -------
        ndarray
            Solved multipliers (parameters). For Ising problem, these can be converted
            into matrix format using utils.vec2mat.
        """

        from scipy import transpose
        
        if reset_rng:
            # return same rng in initial state every time
            rseed = self.model.rng.randint(2**32-1)
            get_rng = lambda rseed=rseed: np.random.RandomState(rseed)
        else:
            get_rng = lambda: self.model.rng

        numDataSamples = len(self.sample)
        # convert data samples to coocMat
        coocMatData = mean_field_ising.cooccurrence_matrix((self.sample+1)/2)
        
        if cooc_cov is None:
            cooc_cov = mean_field_ising.coocSampleCovariance(self.sample)
        
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
            cov = cooc_cov  # / numDataSamples (can't do this here due to numerical issues)
                            # instead include numDataSamples in the calculation of coocMatMeanZSq

        # for use in gammaPrime <-> priorLmbda
        freqsList = np.diag(coocMatData)
        pmean = np.mean(freqsList)
        
        # Generate samples from model (need to translate parameters)
        def samples(J):
           self.model.set_rng(get_rng())
           if min_covariance:
               J = tildeJ2normalJ(J, empiricalFreqs)
           burninDefault = 100*self.n
           J = J + J.T
           self.model.set_multipliers(np.concatenate([J.diagonal(), squareform(zero_diag(-J))]))
           self.model.generate_samples(burninDefault, 1)
           return self.model.sample

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
        J = -zero_diag(J)
        self.multipliers = convert_params( h, squareform(J)*2, '11', concat=True )

        return self.multipliers

    def bracket1d(self, xList, funcList):
        """Assumes xList is monotonically increasing
        
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
