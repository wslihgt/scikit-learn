"""
Factorial Hidden Markov Model

After
[Gha97] Z. Ghahramani and M. Jordan,
Factorial Hidden Markov Models
Machine Learning 29, 245-273 (1997)

Matlab code at:
http://mlg.eng.cam.ac.uk/zoubin/software/fhmm.tar.gz

A few lines coming from scikits.learn.hmm.py

"""
# Factorial Hidden Markov Models
#
# Author: Jean-Louis Durrieu <jean-louis@durrieu.ch>
# 2012


import string
import os 

import numpy as np

# in order to better invert covariance matrices, use of linear algebra:
#     y' C^-1 x is computed by cholesky decomposition of C = L L', then
#     solving the equations L y_ = y and L x_ = x.
#     thereafter: y' C^-1 x = y_' x
from scipy import linalg
# non-negative least square solver, for "smart initialization":
from scipy.optimize import nnls
from scipy.signal import lfilter


import warnings
# from scikits.learn (sklearn) from v0.9:
import sklearn
from .hmm import _BaseHMM, GaussianHMM, decoder_algorithms
from .mixture import (distribute_covar_matrix_to_match_covariance_type,
                      _validate_covars)

if sklearn.__version__.split('.')[1] in ('9','10','8'): #sklearn
    from .utils.extmath import logsum
else:
    from sklearn.utils.extmath import logsumexp as logsum
    ##try:
    ##    from ._hmmc import _logsum as logsum
    ##except ImportError:
    ##    raise ImportError("Problem loading logsum from sklearn._hmmc module")
        

# replace inline by cython, when possible. Here: use current 0.11-dev
# implementation of hmm, in cython.
### home brew forward backward, with inline:
##from scipy.weave import inline
##fwdbwd_filename = 'computeLogDensity_FB_Viterbi.c'
##fwdbwd_file = open(fwdbwd_filename,'r')
##fwdbwd_supportcode = fwdbwd_file.read()
##fwdbwd_file.close()

# for displaying "debugging" information
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

# from scikits.learn.mixture, but without checking the min value:
def normalize(A, axis=None):
    """normaliwe(A, axis=None)
    
    Normalizes the array A by dividing its elements such that
    the sum over the given axis equals 1, except if that sum is 0,
    in which case, it ll stay 0.
    
    Parameters
    ----------
    A : ndarray
        An array containing values, to be normalized.
    axis : integer, optional
        Axis over which the ndarray is normalized. By default, axis
        is None, and the array is normalized such that its sum becomes 1.
    
    Returns
    -------
    out : ndarray
        The normalized ndarray.
    
    See also
    --------
    scikits.learn.normalize

    Examples
    --------
    >>> normalize(np.array([[0., 1.], [0., 5.]]))
    array([[ 0.        ,  0.16666667],
           [ 0.        ,  0.83333333]])
    >>> normalize(np.array([[0., 1.], [0., 5.]]), axis=0)
    array([[ 0.        ,  0.16666667],
           [ 0.        ,  0.83333333]])
    
    """
    Asum = A.sum(axis)
    if not(axis is None) and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    else:
        if Asum==0:
            Asum  = 1
    return A / Asum

# methods for estimation/evaluation/decoding of FHMM:
estimation_methods = ('full', 'variational')

class _BaseFactHMM(_BaseHMM): 
    """ Base class for Factorial HMM
    
    Attributes
    ----------
    n_components : int
        Number of states in the model.
        
    n_components_per_chain : list
        List of number of components for each Markov chain.

    n_chains : int
        Number of Markov chains in the model.

    HMM : list
        List of HMM instances, 
        
    transmat_ : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.
        
    transmat_per_chain : list
        List of matrices of transition probabilities between states,
        one per Markov chain.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    transmat_prior : array, shape (`n_components`, `n_components`)
        Matrix of prior transition probabilities between states.

    startprob_prior : array, shape ('n_components`,)
        Initial state occupation prior distribution.

    algorithm : string, one of the decoder_algorithms
        decoder algorithm

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    See Also
    --------
    hmm : hidden Markov model

    """
    def __init__(self,
                 n_components_per_chain=[1], 
                 startprob_per_chain=None,
                 transmat_per_chain=None,
                 startprob_prior=None,
                 transmat_prior=None,
                 algorithm="viterbi",
                 estimation_method='variational',
                 HMM=None):
        """
        """
        self._n_components_per_chain = list(n_components_per_chain)
        self._n_components = np.prod(self._n_components_per_chain)
        self._n_chains = len(self._n_components_per_chain)
        # number of states, but not all possible states,
        # only sum of number of states for each chain:
        self._n_components_per_chain_all = np.sum(n_components_per_chain)
        
        ## the arguments are distributed to the individual HMMs
        ## of the factorial HMM
        if startprob_per_chain is None or \
               len(startprob_per_chain)!=self._n_chains:
            startprob_per_chain = [None] * self._n_chains
            
        if transmat_per_chain is None or \
               len(transmat_per_chain)!=self._n_chains:
            transmat_per_chain = [None] * self._n_chains
        
        if startprob_prior is None or \
               len(startprob_prior)!=self._n_chains:
            startprob_prior = [None] * self._n_chains
        
        if transmat_prior is None or \
               len(transmat_prior)!=self._n_chains:
            transmat_prior = [None] * self._n_chains        
        
        self.HMM = {}

        if HMM is None or \
            len(HMM) != self._n_chains or \
            np.any([not(isinstance(h, _BaseHMM)) \
                    for n in range(self._n_chains)]) or \
            np.any([HMM[n].n_components != self._n_components_per_chain \
                    for n in range(self._n_chains)]):
            for n in range(self._n_chains):
                self.HMM[n] = _BaseHMM(n_components_per_chain[n],
                                       startprob_per_chain[n],
                                       transmat_per_chain[n],
                                       startprob_prior=startprob_prior[n],
                                       transmat_prior=transmat_prior[n])
        else:
            self.HMM = list(HMM)
        
        if algorithm in decoder_algorithms:
            self._algorithm = algorithm
        else:
            self._algorithm = "viterbi"
        
        if estimation_method in estimation_methods:
            self._estimation_method = estimation_method
        else:
            self._estimation_method = "variational"
        
    def eval(self, obs,
             estimation_method=None,
             **kwargs):
        """Compute the log probability under the model and compute posteriors
        
        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points.  Each row
            corresponds to a single point in the sequence.
            
        Returns
        -------
        logprob : float
            Log likelihood of the sequence `obs`
        posteriors: array_like, shape (n, n_components)
            Posterior probabilities of each state for each
            observation.
            NB: in case method=='variational', the variational approximation
            is used, and the returned posteriors variable is a list of n_chains
            posterior matrices, for chain nc:
                array_like, shape (n, n_components_per_chain[nc])
            
        See Also
        --------
        score : Compute the log probability under the model
        decode : Find most likely state sequence corresponding to a `obs`
        """
        self._set_estimation_method(estimation_method)

        eval_meth = {'full': super(_BaseFactHMM, self).eval,
                     'variational': self.eval_var}
        return eval_meth[self._estimation_method](obs, **kwargs)
    
    def eval_var(self, obs, n_innerLoop=10,
                 tol=0.0001, postInitMeth='random',
                 verbose=False, debug=False):
        """Compute the log probability under the model and compute posteriors
        
        Uses the variational approximation to compute the posterior
        probabilities.
        
        TODO: the returned log-likelihood is not the exact or even its
            approximation. [Gha97] may provide some work-around, if such a
            computation is required.
        
        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points.  Each row
            corresponds to a single point in the sequence.

        n_innerLoop : int
            Number of inner loops (over the n_chains Markov chains) to be done
            to refine the posterior estimation.

        tol : float32
            Tolerance parameter for the inner loop stopping condition

        postInitMethod : string
            Decide the initializing method for the posterior probabilities,
            choose in ('random', 'nmf', 'lms', 'equi').
            See _init_posterior_var.
            
        Returns
        -------
        logprob : float
            Log likelihood of the sequence `obs`
        posteriors: list
            List of n_chains posterior probabilities, one per Markov chain.
            For chain nc:
                array_like, shape (n, n_components_per_chain[nc])
            Posterior probabilities of each state for each
            observation, as approximated by the variational algorithm.
            
        See Also
        --------
        score_var : Compute the log probability under the model
        decode_var : Find most likely state sequence corresponding to a `obs`
        
        """
        obs = np.asanyarray(obs)
        nframes = obs.shape[0]
        
        # initializing the posterior probabilities
        posteriors, logPost = self._init_posterior_var(obs=obs,
                                                       method=postInitMeth,
                                                       debug=debug,
                                                       verbose=verbose)
        
        for i in range(n_innerLoop):
            if verbose:
                print "    inner loop for variational approx.", inn,\
                      "over", n_innerLoop
            # for stopping condition
            posteriors0 = list(posteriors)
            logPost0 = list(logPost)
            # compute variational parameters as in [Gha97]
            # "% first compute h values based on mf"
            # Forward Backward for each chain:
            for n in range(self.n_chains):
                # computing log variational parameters
                if verbose:
                    print "        chain number:", n, "in", self._n_chains
                frameLogVarParams = self._compute_var_params_chain(\
                                                obs=obs,
                                                posteriors=posteriors,
                                                chain=n, debug=debug)
                ## idxStates = np.sum(self.n_components_per_chain[:n])+\
                ##             np.arange(self.n_components_per_chain[n])
                ## idxStates = np.int32(idxStates)
                logprob, fwdLattice = self._do_forward_pass_var_chain(\
                                              frameVarParams=frameLogVarParams,
                                              chain=n,
                                              debug=debug)
                bwdlattice = self._do_backward_pass_var_chain(\
                                              frameVarParams=frameLogVarParams,
                                              chain=n,
                                              debug=debug)
                gamma = fwdlattice + bwdlattice
                logPost[n] = (gamma - logsum(gamma, axis=1)).T
                posteriors[n] = np.exp(logPost[n])
            
            delmf = np.sum(np.sum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost.values(),axis=1)))- \
                    np.sum(np.sum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost0.values(),axis=1)))
            
            if delmf < nframes*tol and i>0:
                itermf = i
                break
        
        return frameLogVarParams, posteriors # approx. of loglikelihood?
    
    def decode(self, obs, estimation_method='variational', **kwargs):
        """returns most likely seq of states, and
        the posterior probabilities.
        """
        self._set_estimation_method(estimation_method)
        
        decode_meth = {'full': super(_BaseFactHMM, self).decode,
                       'variational': self.decode_var}
        return decode_meth[self._estimation_method](obs, **kwargs)
    
    def decode_var(self, obs, n_innerLoop=10,
                   verbose=False, debug=False,
                   n_repeatPerChain=None, postInitMeth='random'):
        """decode_var
        
        returns most likely seq of states, and
        the posterior probabilities.
        
        For now, the sequence of states is calculated through the
        variational approximation, i.e. we first compute the variational
        parameters and the corresponding posteriors for each chain, iterate
        over that process n_innerLoop times, and at last outputs the states
        computed through Viterbi for each chain, on the log variational
        parameters as \"log likelihood\" (as suggested in [Gha97]).
        
        Important I : this is not the true sequence (should run Viterbi on all
        states, with true likelihood). It is possible to do, but very expensive
        in terms of memory (especially for source/filter subclass).
        
        """
        obs = np.asanyarray(obs)
        
        nframes = obs.shape[0]
        
        if n_repeatPerChain is None:
            n_repeatPerChain = [1] * self._n_chains
        if len(n_repeatPerChain) != self._n_chains:
            print "Warning: the provided n_repeatPerChain arg has wrong length."
            print "         Falling to default: only one repeat for each chain."
            n_repeatPerChain = [1] * self._n_chains
        
        ## First compute the variational parameters
        if verbose:
            print "Computing the variational parameters"
        
        posteriors, logPost = self._init_posterior_var(obs=obs,
                                                       method=postInitMeth,
                                                       debug=debug,
                                                       verbose=verbose)
        if debug:
            plt.figure(200)
            plt.clf()
            plt.imshow(np.concatenate(logPost.values(), axis=1).T)
            plt.colorbar()
            plt.title('initial log posteriors')
            plt.draw()
        
        for inn in range(n_innerLoop):
            if verbose:
                print "    inner loop for variational approx.", inn,\
                      "over", n_innerLoop
            # for stopping condition
            posteriors0 = dict(posteriors)
            logPost0 = dict(logPost)
            for n in range(self._n_chains):
                for t in range(n_repeatPerChain[n]):
                    # computing log variational parameters
                    if verbose:
                        print "        chain number:", n, "in", self._n_chains
                    frameLogVarParams = self._compute_var_params_chain(\
                                                obs=obs,
                                                posteriors=posteriors,
                                                chain=n, debug=debug)
                    # computing the forward/backward passes on the variational
                    # parameters, for the given chain
                    logprob, fwdlattice = self._do_forward_pass_var_chain(\
                                              frameVarParams=frameLogVarParams,
                                              chain=n,
                                              debug=debug)
                    bwdlattice = self._do_backward_pass_var_chain(\
                                              frameVarParams=frameLogVarParams,
                                              chain=n,
                                              debug=debug)
                    # ... and deducing the posterior probas
                    gamma = fwdlattice + bwdlattice
                    logPost[n] = (gamma.T - logsum(gamma, axis=1)).T
                    posteriors[n] = np.exp(logPost[n])
                    
                    if debug:
                        plt.figure(1)
                        plt.clf()
                        plt.imshow(frameLogVarParams-\
                                   np.vstack(logsum(frameLogVarParams,axis=1)))
                        plt.colorbar()
                        plt.title("frameLogVarParams "+str(n))
                        plt.draw()
                        plt.figure(2)
                        plt.clf()
                        plt.subplot(211)
                        plt.imshow(posteriors[n].T)
                        plt.colorbar()
                        plt.title("posterior "+str(n))
                        plt.subplot(212)
                        plt.imshow(bwdlattice.T)
                        plt.colorbar()
                        plt.title("bwdlattice "+str(n))
                        plt.draw()
                        
                        # print 'n',n,'frameLogVarParams',\
                        #       np.any(np.isnan(frameLogVarParams)),\
                        #       'fwd',np.any(np.isinf(fwdlattice[n])),\
                        #       'bwd',np.any(np.isinf(bwdlattice[n]))
                        ## raw_input("press \'any\' key... Doh!")
                        
            # stopping condition (from [Gha97] and Matlab code)
            # measuring relative difference between previous and current
            # posterior probabilities.
            delmf = np.sum(np.sum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost.values(),axis=1)))- \
                    np.sum(np.sum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost0.values(),axis=1)))
            
            # TODO: check this stopping condition
            if delmf < nframes*0.000001 and inn>0:
                itermf = inn
                break
            
        # then use "stabilized" variational params to compute
        # the state sequences:
        state_sequences = {}
        ## frameVarParams = self._compute_var_params(obs, posteriors)
        # print frameVarParams #DEBUG
        for n in range(self._n_chains):
            frameLogVarParams = self._compute_var_params_chain(\
                                                obs=obs,
                                                posteriors=posteriors,
                                                chain=n, debug=debug)
            logprob, state_sequences[n] = self.HMM[n]._do_viterbi_pass(\
                                              frameLogVarParams)
            del logprob
            
        # TODO return a meaningful logprob value:
        logprob = None
        return logprob, state_sequences, posteriors

    def fit(self, obs, estimation_method='variational', **kwargs):
        """fit(obs,
        
        Arguments
        
        See also:
        fit_var: variational approach to fit the FHMM parameters to obs
        """
        self._set_estimation_method(estimation_method)
        
        fit_meth = {'full': super(_BaseFactHMM, self).fit,
                    'variational': self.fit_var}
        
        return fit_meth[self._estimation_method](obs, **kwargs)
    
    def fit_var(self, obs, n_iter=100, n_innerLoop=10,
                thresh=1e-2, params=string.letters,
                init_params=string.letters,
                tol=0.0001, innerTol=1e-6,
                verbose=False, debug=False,
                postInitMeth='random',
                **kwargs):
        """fit_var
        
        n_iter is number of baum-welch cycles
        n_innerLoop number of mean field computations (struct. var. approx.)
        
        """
        self._init(obs, init_params)
        
        logprob = []
        for i in xrange(n_iter):
            if verbose:
                print "iteration", i,"over",n_iter
            # Expectation step
            stats = self._initialize_sufficient_statistics_var()
            curr_logprob = 0
            for seq in obs:
                nframes = seq.shape[0]
                
                fwdlattice = {}
                bwdlattice = {}

                posteriors, logPost = self._init_posterior_var(obs=seq,
                                                               method=postInitMeth,
                                                               debug=debug,
                                                               verbose=verbose)
                
                for inn in range(n_innerLoop):
                    if verbose:
                        print "    inner loop for variational approx.", inn,\
                              "over",n_innerLoop
                    # for stopping condition
                    posteriors0 = dict(posteriors)
                    logPost0 = dict(logPost)
                    ## the following commented version works well for small
                    ## scale problems, but with big _n_components_per_chain, one should
                    ## consider not storing all the variational params.
                    ## # compute variational parameters as in [Gha97]
                    ## frameVarParams = self._compute_var_params(seq, posteriors)
                    ## # with inline :
                    ## fwdlattice, bwdlattice = self._do_fwdbwd_pass_var_c(\
                    ##                                           frameVarParams)
                    for n in range(self._n_chains):
                        ## idxStates = np.sum(self.n_components_per_chain[:n])+\
                        ##             np.arange(self.n_components_per_chain[n])
                        ## idxStates = np.int32(idxStates)
                        ## dumpLogProba, fwdlattice[n] = \
                        ##     self.HMM[n]._do_forward_pass(\
                        ##                              frameVarParams[n],
                        ##                              maxrank,
                        ##                              beamlogprob)
                        ## bwdlattice[n] = self.HMM[n]._do_backward_pass(\
                        ##                                     frameVarParams[n],
                        ##                                     fwdlattice[n],
                        ##                                     maxrank,
                        ##                                     beamlogprob)
                        # 20111219T1725 change of plans: compute one by one
                        # otherwise takes too much memory
                        frameLogVarParams = self._compute_var_params_chain(\
                                                obs=seq,
                                                posteriors=posteriors,
                                                chain=n)
                        ##fwdlattice[n], bwdlattice[n] = \
                        ##                        self._do_fwdbwd_pass_var_hmm_c(\
                        ##                                         frameLogVarParams,
                        ##                                         chain=n)
                        dumpLogProba, fwdlattice[n] = \
                            self._do_forward_pass_var_chain(frameLogVarParams,
                                                            chain=n)
                        bwdlattice[n] = \
                            self._do_backward_pass_var_chain(frameLogVarParams,
                                                             chain=n)
                        
                        if debug:
                            plt.figure(1)
                            plt.clf()
                            plt.imshow(frameLogVarParams)
                            plt.title("frameLogVarParams")
                            plt.draw()
                            plt.figure(2)
                            plt.clf()
                            plt.subplot(211)
                            plt.imshow(fwdlattice[n])
                            plt.title("fwdlattice")
                            plt.subplot(212)
                            plt.imshow(bwdlattice[n])
                            plt.title("bwdlattice")
                            plt.draw()
                        
                        gamma = fwdlattice[n] + bwdlattice[n]
                        logPost[n] = (gamma.T - logsum(gamma, axis=1)).T
                        posteriors[n] = np.exp(logPost[n])
                        ## del dumpLogProba
                    ## lpr, fwdlattice = self._do_forward_pass_var(frameVarParams,
                    ##                                             maxrank,
                    ##                                             beamlogprob)
                    ## bwdlattice = self._do_backward_pass(framelogprob, fwdlattice,
                    ##                                     maxrank, beamlogprob)
                    ## gamma = fwdlattice + bwdlattice
                    ## posteriors = np.exp(gamma.T - logsum(gamma, axis=1)).T
                    
                    # Calculate condition to stop inner loop
                    #     from pmhmm.m of [Gha97]
                    delmf = np.sum(np.sum(np.concatenate(posteriors.values(),
                                                         axis=1)*\
                                        np.concatenate(logPost.values(),
                                                       axis=1)))- \
                            np.sum(np.sum(np.concatenate(posteriors.values(),
                                                         axis=1)*\
                                        np.concatenate(logPost0.values(),
                                                       axis=1)))
                    ## print delmf#DEBUG
                    if delmf < nframes*0.000001 and inn>0:
                        itermf = inn
                        break
                
                # curr_logprob += lpr
                self._accumulate_sufficient_statistics_var_chain(
                    stats, seq, posteriors, fwdlattice,
                    bwdlattice, params)
                # logprob.append(curr_logprob)
            
            # Check for convergence.
            # TODO: compute true likelihood to take decision
            ## if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
            ##     break

            # correcting accumulated stats
            for n in range(self._n_chains):
                n0 = np.sum(self.n_components_per_chain[:n])
                n1 = n0 + self.n_components_per_chain[n]
                #stats['cor'][np.ix_(idxStates,idxStates)]
                stats['cor'][n0:n1,n0:n1] = np.diag(stats[n]['post'])
                
            stats['cor'] = 0.5 * (stats['cor'] + stats['cor'].T)
            
            # at this point, links with [Gha97]:
            #     (especially for subclass GaussianFHMM)
            #     stats['obs'] -> GammaX
            #     stats['post'] -> gammasum
            #     stats['cor'] -> Eta
            #     stats['trans'] -> Xi
            
            # Maximization step
            self._do_mstep(stats, params, **kwargs)
            
        return self
    
    def _init(self, obs, params):
        if (hasattr(self, 'n_features')
            and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        
        self.n_features = obs[0].shape[1]
        
        for n in range(self._n_chains):
            self.HMM[n]._init(obs, params)

        self.startprob = None
        self.transmat = None

    def _set_estimation_method(self, estimation_method):
        if not(hasattr(self, '_estimation_method')) or \
              estimation_method is not None:
            if not(estimation_method in estimation_methods):
                warn("Desired estimation method should be in "+\
                     estimation_methods+\
                     ",\nfalling back on default variational method.")
                self._estimation_method = "variational"
            else:
                self._estimation_method = estimation_method
        
    @property
    def n_chains(self):
        """Number of states in the model."""
        return self._n_chains
    
    @property
    def n_components(self):
        """Number of states in the full HMM model."""
        return self._n_components
    
    @property
    def n_components_per_chain(self):
        """Number of states in the model."""
        return self._n_components_per_chain
    @property
    def n_components_per_chain_all(self):
        """Number of states in the model."""
        return self._n_components_per_chain_all

    # FHMM equivalent to HMM with prod(n_components_per_chain), e.g. for each chain n,
    # if n_components_per_chain[n]=K, then the total n_components is K**(n_chains)
    # 
    # It is preferred not to store all the corresponding transition and
    # start probabilities, computing them only if required by the user.
    #
    # TODO: check that these are not called internally by some function
    # in "general" hmm module; override such methods (especially: fit, eval, decode...).
    #
    # NB: these computations do not prevent over/underflows, since the goal
    # of an FHMM implementation is not to use it as a regular HMM.
    # It may anyway break due to memory issues (growing exponentially in number
    # of chains)
    
    # startprob are properties, because you cant trivially factorize them
    # directly from the full HMM startproba:
    @property
    def startprob_(self):
        startprob_ = self.HMM[0].startprob_
        for n in range(1,self._n_chains):
            startprob_ = np.kron(startprob_, self.HMM[n].startprob_)
        return startprob_
        
    @property
    def _log_startprob(self):
        return np.log(self.startprob_)
    
    # transmat_per_chain as property:
    def _get_transmat_per_chain(self):
        """the transition matrices as a list of the HMM transition matrices
        """
        return [self.HMM[n].transmat_ for n in range(self._n_chains)]
    
    def _set_transmat_per_chain(self, transmat):
        if len(transmat) != self._n_chains:
            raise ValueError("Length of transmat list should equal the "+\
                             "number of Markov chains.")
        else:
            for n in range(self._n_chains):
                self.HMM[n].transmat_ = transmat[n]
    
    transmat_per_chain_ = property(_get_transmat_per_chain,
                                   _set_transmat_per_chain)
    
    # startprob_per_chain as property:
    def _get_startprob_per_chain(self):
        """the transition matrices as a list of the HMM transition matrices
        """
        return [self.HMM[n].startprob_ for n in range(self._n_chains)]
    
    def _set_startprob_per_chain(self, startprob):
        if len(startprob) != self._n_chains:
            raise ValueError("Length of startprob list should equal the "+\
                             "number of Markov chains.")
        else:
            for n in range(self._n_chains):
                self.HMM[n].startprob_ = startprob[n]
    
    startprob_per_chain_ = property(_get_startprob_per_chain,
                                   _set_startprob_per_chain)
    
    # _log_transmat and transmat_ as property:
    @property
    def transmat_(self):
        """The actual transmat_ for the full HMM is given as
        a prod(n_components_per_chain) x prod(n_components_per_chain) matrix.
        
        
        """
        # transmat_ = np.zeros(self.n_components, self.n_components) # final shape
        # using Kronecker products to set the whole transition matrix:
        transmat_ = self.HMM[0].transmat_
        for n in range(1, self._n_chains):
            transmat_ = np.kron(transmat_, self.HMM[n].transmat_)
            
        # NB: accessing the desired transition probability
        # is therefore related to the kronecker product properties:
        # here, the states are such that, for state i_n and j_n of chain n, the 
        # corresponding transition probability is given in transmat_[i,j]
        # where
        #    i = \sum_n (\sum_{m=n+1}^{n_chains}n_components_per_chain[m] i_n),
        # and likewise for j.
        return transmat_
    
    @property
    def _log_transmat(self):
        """The actual _log_transmat for the full HMM is given as
        a prod(n_components_per_chain) x prod(n_components_per_chain) matrix.
        
        
        """
        return np.log(self.transmat_)
    # should not set transmat or log_transmat manually:
    #def _set__log_transmat(self, log_transmat):
    #    pass   
    
    
    def _initialize_sufficient_statistics_var(self):
        stats = {}
        for n in range(self._n_chains):
            stats[n] = self.HMM[n]._initialize_sufficient_statistics()
        
        # specific to FHMM: correlation expectation
        #    [Gha97] : the sum_t < S_t S_t'>
        stats['cor'] = np.zeros([self._n_components_per_chain_all,
                                 self._n_components_per_chain_all])
        return stats
    
    def _accumulate_sufficient_statistics_var(self, stats, seq,
                                              frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """distribute the results and compute sufficient statistics
        for each chain
        
        
        """
        for n in range(self.n_chains):
            self.HMM[n]._accumulate_sufficient_statistics(\
                stats[n], seq, frameVarParams[n],
                posteriors[n], fwdlattice[n],
                bwdlattice[n], params)
        
        # specific to FHMM: first order correlation expectation
        #    [Gha97] : sum_t < S_t S_t'>
        postTot = np.concatenate(posteriors.values(), axis=1)
        stats['cor'] += np.dot(postTot.T, postTot)
        del postTot
    
    def _accumulate_sufficient_statistics_var_chain(self, stats, seq,
                                                    posteriors, fwdlattice,
                                                    bwdlattice, params):
        """distribute the results and compute sufficient statistics
        for each chain
        
        computes the estimates of the variational parameters
        internally (avoiding to store it as it may exceed memory
        capacity).
        """
        for n in range(self.n_chains):
            frameLogVarParams = self._compute_var_params_chain(seq,
                                                     posteriors=posteriors,
                                                     chain=n)
            self.HMM[n]._accumulate_sufficient_statistics(\
                stats[n], seq, frameLogVarParams,
                posteriors[n], fwdlattice[n],
                bwdlattice[n], params)
        
        # specific to FHMM: first order correlation expectation
        #    [Gha97] : the sum_t < S_t S_t'>
        postTot = np.concatenate(posteriors.values(), axis=1)
        # TODO: to be checked, the diagonals blocks are probably wrong here:
        # (they are equal to diag(<S_t^n>), for block/chain n)
        stats['cor'] += np.dot(postTot.T, postTot)
        del postTot

    def _init_posterior_var(self, obs, method='random', **kwargs):
        """initialize the posterior probabilities with desired method:
            'random'
            'equi'
            'nmf'
            'lms'
        """
        init_fun_dict = {'random': self._init_posterior_var_random,
                         'equi'  : self._init_posterior_var_equi,
                         'nmf'   : self._init_posterior_var_nmf,
                         'lms'   : self._init_posterior_var_lms}
        return init_fun_dict[method](obs=obs, **kwargs)
    
    def _init_posterior_var_random(self, obs, verbose=False, debug=False):
        """random posterior probabilities
        """
        nframes = obs.shape[0]
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains): 
            posteriors[n] = np.random.rand(nframes,
                                           self._n_components_per_chain[n])
            posteriors[n] = normalize(posteriors[n], axis=1)
            logPost[n] = np.log(posteriors[n])
        return posteriors, logPost
    
    
    def _init_posterior_var_equi(self, obs, verbose=False, debug=False):
        """equiprobable posterior probabilities
        """
        nframes = obs.shape[0]
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains):
            posteriors[n] = np.ones([nframes, self._n_components_per_chain[n]])/\
                            (1.*self._n_components_per_chain[n])
            logPost[n] = np.log(posteriors[n])
        return posteriors, logPost
    
    
    def _init_posterior_var_nmf(self, obs, verbose=False, debug=False):
        """posteriors as normalized nmf amplitude coefficients,
        if it makes sense (for a given observation likelihood)
        """
        pass
    
    def _init_posterior_var_lms(self, obs, verbose=False, debug=False):
        """posteriors as normalized lms solution,
        if all components allowed everywhere, at once. 
        """
        pass
    
    def _do_forward_pass_var(self, frameVarParams, **kwargs):
        """run forward pass, assuming the variational approximation
        that each HMM is isolated, and for each chain n, the corresponding
        'observation likelihood' is given by frameVarParams[n]
        """
        logprob = {}
        fwdlattice = {}
        for n in range(self.n_chains):
            logprob[n], fwdlattice[n] = self.HMM[n]._do_forward_pass(\
                                                           frameVarParams[n],
                                                           **kwargs)
        
        return logprob, fwdlattice
    
    def _do_backward_pass_var(self, frameVarParams, **kwargs):
        """run backward pass, assuming the variational approximation
        that each HMM is isolated, and for each chain n, the corresponding
        'observation likelihood' is given by frameVarParams[n]
        """
        bwdlattice = {}
        for n in range(self.n_chains):
            bwdlattice[n] = self.HMM[n]._do_backward_pass(frameVarParams[n],
                                                          **kwargs)
        return bwdlattice
    
    def _do_forward_pass_var_chain(self, frameVarParams, chain, **kwargs):
        """run forward pass, assuming the variational approximation
        that each HMM is isolated, and for each chain n, the corresponding
        'observation likelihood' is given by frameVarParams[n]
        """
        return self.HMM[chain]._do_forward_pass(frameVarParams)
    
    def _do_backward_pass_var_chain(self, frameVarParams, chain, **kwargs):
        """run backward pass, assuming the variational approximation
        that each HMM is isolated, and for the given chain, the corresponding
        'observation likelihood' is given by frameVarParams
        """
        return self.HMM[chain]._do_backward_pass(frameVarParams)
    
    def _do_fwdbwd_pass_var_hmm_c_deprecated(self, frameLogVarParams, chain):
        """n is the chain number
        
        here, frameLogVarParams are the estimated variational parameters, for the given
        chain 'chain'.
        
        deprecated: from scipy 0.11-dev, use cython implementation from hmm module
        """
        nframes = frameLogVarParams.shape[0]
        n = chain
        
        fwdbwd_callLine = "computeForwardBackward(nframes, n_components_per_chain, "+\
                          "startprob, transmat,"+\
                          "dens, logdens, logAlpha, logBeta, alpha, beta);"
        
        
        if not frameLogVarParams[n].flags['C_CONTIGUOUS']:
            raise ValueError("FrameVarParams should be C contiguous!")
        fwdloglattice = np.zeros([nframes, self._n_components_per_chain[n]])
        bwdloglattice = np.zeros([nframes, self._n_components_per_chain[n]])
        fwdlattice = np.zeros([nframes, self._n_components_per_chain[n]])
        bwdlattice = np.zeros([nframes, self._n_components_per_chain[n]])
        inline(fwdbwd_callLine,
               arg_names=['nframes', 'n_components_per_chain', \
                          'startprob', 'transmat', \
                          'dens', 'logdens', \
                          'logAlpha', 'logBeta', 'alpha', 'beta'],
               local_dict={'nframes': nframes,
                           'n_components_per_chain': self._n_components_per_chain[n],
                           'startprob': self.HMM[n].startprob,
                           'transmat': self.HMM[n].transmat,
                           'dens': np.array([0.]),# unused argument
                           'logdens': frameLogVarParams,
                           'logAlpha': fwdloglattice,
                           'logBeta': bwdloglattice,
                           'alpha': fwdlattice,
                           'beta': bwdlattice},
               support_code=fwdbwd_supportcode,
               extra_compile_args =['-O3 -fopenmp'], \
               extra_link_args=['-O3 -fopenmp'], \
               force=0, verbose=2)

        ##print frameLogVarParams #DEBUG
        ##print fwdlattice #DEBUG
        ##print fwdloglattice #DEBUG
        
        del fwdlattice
        del bwdlattice
        
        return fwdloglattice, bwdloglattice
    
    def _do_fwdbwd_pass_var_c_deprecated(self, frameLogVarParams):
        nframes = frameLogVarParams[0].shape[0]
        
        fwdbwd_callLine = "computeForwardBackward(nframes, n_components_per_chain, "+\
                          "startprob, transmat, "+\
                          "dens, logdens, logAlpha, logBeta, alpha, beta);"
        fwdloglattice = {}
        bwdloglattice = {}
        
        for n in range(self._n_chains):
            fwdloglattice[n], bwdloglattice[n] = \
                                           self._do_fwdbwd_pass_var_hmm_c_deprecated(\
                                                    frameLogVarParams[n], n)
        
        return fwdloglattice, bwdloglattice
    
    def _compute_var_params(self, obs, posteriors):
        pass
    
    def _compute_var_params_chain(self, obs, posteriors, chain, debug=False):
        pass
    
    def _compute_log_likelihood(self, obs):
        pass
    
    def _do_mstep(self,stats,params,**kwargs):
        for n in range(self._n_chains):
            ## print params # DEBUG
            self.HMM[n]._do_mstep(stats[n], params, **kwargs)

class GaussianFHMM(_BaseFactHMM, GaussianHMM): # should subclass also GaussianHMM
    """Class implementing factorial HMM
    
    Assuming conditional Gaussian emission model where the mean is the sum
    over the Markov chains of the means of the 'active' states, and the
    covariance matrix does not depend on the states.
    
    """
    
    def __init__(self, covariance_type='tied',
                 means_prior=None, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 n_components_per_chain=[2,2],
                 startprob_per_chain=None,
                 transmat_per_chain=None,
                 startprob_prior=None,
                 transmat_prior=None,
                 HMM=None,
                 estimation_method='variational',
                 algorithm='viterbi'):
        """
        """
        super(GaussianFHMM, self).__init__(n_components_per_chain=\
                                           n_components_per_chain, 
                                           startprob_per_chain=\
                                           startprob_per_chain,
                                           transmat_per_chain=\
                                           transmat_per_chain,
                                           startprob_prior=startprob_prior,
                                           transmat_prior=transmat_prior,
                                           estimation_method=estimation_method,
                                           algorithm=algorithm)
        
        if startprob_per_chain is None or \
               len(startprob_per_chain)!=self._n_chains:
            startprob_per_chain = [None] * self._n_chains
            
        if transmat_per_chain is None or \
               len(transmat_per_chain)!=self._n_chains:
            transmat_per_chain = [None] * self._n_chains
        
        if startprob_prior is None or \
               len(startprob_prior)!=self._n_chains:
            startprob_prior = [None] * self._n_chains
        
        if transmat_prior is None or \
               len(transmat_prior)!=self._n_chains:
            transmat_prior = [None] * self._n_chains
        
        
        # self.HMM[n] should actually be a GaussianHMM, for this model
        # because it is characterized by its mean
        # That's gonna make the method _accumulate_sufficient_statistics
        # keep the running mean for each state of each chain
        self.HMM = {}
        for n in range(self._n_chains):
            self.HMM[n] = GaussianHMM(n_components=\
                                      n_components_per_chain[n],
                                      covariance_type=covariance_type,
                                      startprob=startprob_per_chain[n],
                                      transmat=transmat_per_chain[n],
                                      startprob_prior=startprob_prior[n],
                                      transmat_prior=transmat_prior[n],
                                      means_prior=means_prior,
                                      means_weight=means_weight,
                                      covars_prior=covars_prior,
                                      covars_weight=covars_weight)
        
        self._cvtype = covariance_type
        if not self._cvtype in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad cvtype')
        if self._cvtype is not 'tied' and \
               self._estimation_method is 'variational':
            warnings.warn(
                "Variational estimation method only suitable for tied\n"+
                "covariance matrix. Setting cvtype to \"tied\".")
            self._cvtype='tied'
        
        self.means_prior = means_prior
        self.means_weight = means_weight
        
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
    
    def decode_var(self, obs, n_innerLoop=10,
                   verbose=False, debug=False,
                   n_repeatPerChain=None, **kwargs):
        """decode_var
        
        returns most likely output mean, the estimated seq of states, and
        the posterior probabilities.
        
        For now, the sequence of states is calculated through the
        variational approximation, i.e. we first compute the variational
        parameters and the corresponding posteriors for each chain, iterate
        over that process n_innerLoop times, and at last outputs the states
        computed through Viterbi for each chain, on the log variational
        parameters as \"log likelihood\" (as suggested in [Gha97]).

        At last, we compute the mean corresponding to those sequences.
        
        Important: this is not the true sequence (should run Viterbi on all
        states, with true likelihood). It is possible to do, but very expensive
        in terms of memory (especially for source/filter subclass).
        
        """
        obs = np.asanyarray(obs)
        
        nframes = obs.shape[0]
        
        logprob, state_sequences, posteriors = \
            super(GaussianFHMM, self).decode_var(obs=obs,
                                                 n_innerLoop=n_innerLoop,
                                                 verbose=verbose,
                                                 debug=debug,
                                                 n_repeatPerChain=\
                                                 n_repeatPerChain,
                                                 **kwargs)
        
        # compute outputmeans :
        outputMean = np.zeros([nframes,self.n_features])
        for n in range(self._n_chains):
            outputMean += self.means[n][state_sequences[n]]
        
        return outputMean, state_sequences, posteriors

    #def _init(self, obs, params='stmc', estimation_method="variational"):
    #    """initialize parameters
    #    If estimation method is variational, then necessary to call specific 
    #    """
    #    self._set_estimation_method(estimation_method=estimation_method)
    #    init_methods = {
    #        'full': super(GaussianFHMM,self)._init,
    #        'variational': self._init_var}
    #    return init_methods[self._estimation_method](obs, params=params)
    
    # 20120323 JLD commented, to be removed?
    #    EDIT: no, keep it, because with variational approximation, different
    #    definition of means and covars
    def _init(self, obs, params='stmc'):
        # this calls the _BaseFactHMM._init method
        super(GaussianFHMM, self)._init(obs, params=params)
        
        if (hasattr(self, 'n_features')
            and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        
        self.n_features = obs[0].shape[1]
        
        # Initialize like [Gha97]
        if 'c' in params:
            cv = np.cov(np.concatenate(obs),rowvar=0)
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._cvtype, self._n_components)
        
        if 'm' in params:
            self._means = {}
            for n in range(self._n_chains):
                # TODO: if cvtype not full, then probably issue here!
                self._means[n] = \
                    np.dot(np.random.randn(self._n_components_per_chain[n],
                                           self.n_features), \
                           np.double(linalg.sqrtm(self.covars_[0])))/\
                           self._n_chains + \
                           np.concatenate(obs).mean(axis=0) / \
                           self._n_chains
    
    def _set_estimation_method(self, estimation_method):
        super(GaussianFHMM,self)._set_estimation_method(estimation_method=\
                                                        estimation_method)
        if self._cvtype is not 'tied' and \
               self.estimation_method is 'variational':
            warnings.warn(
                "Variational estimation method only suitable for tied\n"+
                "covariance matrix. Setting cvtype to \"tied\".")
            self._cvtype='tied'

    @property
    def _covariance_type(self):
        return self._cvtype
    
    @property
    def cvtype(self):
        """Covariance type of the model.
        
        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._cvtype
    
    # means of corresponding "regular" hmm, as a property:
    @property
    def _means_(self):
        # NB: accessing the desired mean
        # is therefore related to the kronecker product properties:
        # here, the states are such that, for state i_n and j_n of chain n, the 
        # corresponding mean is given in means[i,j]
        # where
        #    i = \sum_n (\sum_{m=n+1}^{n_chains}n_components_per_chain[m] i_n),
        # and likewise for j.
        means = np.zeros([self._n_components,
                          self.n_features])
        for n in range(self._n_chains):
            idxc = np.arange(self._n_components_per_chain[n])
            idx = np.kron(
                      np.kron(
                          np.ones(np.prod(self._n_components_per_chain[:n])),
                          idxc),
                      np.ones(np.prod(self._n_components_per_chain[(n+1):])))
            means += self._means[n][np.int32(idx)]
        return means
    
    # same for the covariances:
    # 20120323 JLD no need for this if cvtyp eis tied, even for variational method
    #def _get__covars_(self):
    #    """Because the _covars_, normally the internal value, has to
    #    be created from the _covars_ of each individual HMMs
    #    """
    #    return self._covars
    #
    #def _set__covars_(self, covars):
    #    self._covars = covars
    #
    #_covars_ = property(_get__covars_, _set__covars_)

    # means as list of means for each HMM chain:
    def _get_means(self):
        """Mean parameters for each state."""
        return self._means
    
    def _set_means(self, means):
        """setting the means of each chain: must be list of len==_n_chains
        """
        if len(means)!=self._n_chains:
            raise ValueError('means must be a list of length n_chains')
        
        for n in range(self._n_chains):
            if hasattr(self, 'n_features') and \
                   means[n].shape != (self._n_components_per_chain[n], self.n_features):
                raise ValueError('means must have shape (n_components_per_chain, n_features)')
            if means[n].shape[0] != self._n_components_per_chain[n]:
                raise ValueError('means must have shape (n_components_per_chain, n_features)')
        
        self._means = list(means)
        self.n_features = self._means[0].shape[1]
    
    means = property(_get_means, _set_means)
    means_per_chain = property(_get_means, _set_means)
    
    # same for the covariance, for each HMM chain
    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.cvtype == 'full':
            return self._covars_
        elif self.cvtype == 'diag':
            return [np.diag(cov) for cov in self._covars_]
        elif self.cvtype == 'tied':
            return [self._covars_] * self._n_components
        elif self.cvtype == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars_]
    
    def _set_covars(self, covars):
        """[Gha97] : covar matrix unique for all possible states"""
        covars = np.asanyarray(covars)
        # [Gha97] model with only one covar matrix for observation likelihood:
        _validate_covars(covars, self._cvtype, 1, self.n_features)
        self._covars_ = covars.copy()
    
    covars = property(_get_covars, _set_covars)
    
    def _compute_var_params(self, obs, posteriors):
        
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        frameVarParams={}
        # actually frameLogVarParams !
        
        # print self._covars_#DEBUG
        L = linalg.cholesky(self._covars_, lower=True)
        for n in range(self._n_chains):
            cv_sol = linalg.solve_triangular(L, self._means[n].T, lower=True).T
            deltas = np.sum(cv_sol ** 2, axis=1) # shape=(_n_components_per_chain[n])
            obs_sol = linalg.solve_triangular(L,
                                              (obs - obsHat + \
                                               sigHatPerChain[n]).T,
                                              lower=True)
            frameVarParams[n] = np.dot(obs_sol.T, cv_sol.T) - \
                                0.5 * deltas # shape=(nframes, _n_components_per_chain[n])
        
        return frameVarParams
    
    def _compute_var_params_chain(self, obs, posteriors, chain, debug=False):
        
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        ##print self._covars[0]#DEBUG
        n = chain
        # the following is the same for all HM chains: should factorize
        L = linalg.cholesky(self._covars_, lower=True)
        
        cv_sol = linalg.solve_triangular(L, self._means[n].T, lower=True).T
        deltas = np.sum(cv_sol ** 2, axis=1) # shape=(_n_components_per_chain[n])
        obsHatChain = obsHat - sigHatPerChain[n]
        if hasattr(self, 'noiseLevel'):
            obsHatChain = np.maximum(obsHatChain, self.noiseLevel)
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        
        return np.dot(obs_sol.T, cv_sol.T) - \
                         0.5 * deltas # shape=(nframes, _n_components_per_chain[n])
    
    def _compute_log_likelihood(self, obs):
        # return lmvnpdf(obs, self._means, self._covars, self._cvtype)
        pass
    
    def _xH_invC_x_chol(self, X):
        """returns np.dot(np.dot(x_n.T, inv(covar)), x_n),
        using cholesky decomposition of covar
        
        code inspired from scikits.learn.mixture
        
        used to compute Delta^(m) [Gha97] or could be...
        """
        nframes, nfeats = X.shape
        if nfeats!=self.n_features:
            raise ValueError("number of features should be n_features")
        L = linalg.cholesky(self._covars_)
        cv_sol = linalg.solve_triangular(L, X.T, lower=True).T
        
        return np.sum(cv_sol ** 2, axis=1)
    
    def _do_mstep(self, stats, params=string.letters,
                  paramsHMMs = 'st', **kwargs):
        # this is going to update the priors and transitions in self.HMM
        #     avoid that it updates mean and cov (because they re GaussianHMM)
        
        super(GaussianFHMM, self)._do_mstep(stats, paramsHMMs, **kwargs)
        
        # compute SVD instead of inverse of stats['cor'] <-> Eta
        stats['cor'] = 0.5*(stats['cor'].T+stats['cor'])
        
        if 'm' in params or 'c' in params:
            U,s,V = linalg.svd(stats['cor'])
            Si = np.zeros([self._n_components_per_chain_all,
                           self._n_components_per_chain_all])
            
            for n in range(self._n_components_per_chain_all):
                if s[n] >= (s.size * linalg.norm(s) * 0.001):
                    Si[n,n] = 1./s[n]
            
            # just to be sure:
            # print stats['cor'], np.dot(np.dot(U,np.diag(s)),V)#DEBUG
            ## print V.T.conj(), Si, U.T.conj(), stats['cor']#DEBUG
            ## print np.dot(V.T.conj(),V) #DEBUG OK
            ## print np.dot(U.T.conj(),U) #DEBUG OK
            ## print s #DEBUG 
            ## print np.dot(np.dot(V.T.conj(),np.diag(1./s)),np.dot(U.T.conj(),stats['cor']))#DEBUG - for some reason, this is *not* identity... which is annoying... probably because one of the singular values is small ?
            
            # update the means:
            GammaX = np.concatenate([stats[n]['obs'] \
                                     for n in range(self._n_chains)])
            
            ## print Si, GammaX #DEBUG
            ## print V,s,S,U, GammaX #DEBUG
        
        if 'm' in params:
            newMean = np.dot(np.dot(V.T.conj(), Si), np.dot(U.T.conj(), GammaX))
            ## print newMean
            ## print newMean #DEBUG
            for n in range(self._n_chains):
                n0 = np.sum(self.n_components_per_chain[:n])
                n1 = n0 + self.n_components_per_chain[n]
                self._means[n] = newMean[n0:n1]
        
        ## print self.means
        
        # update covariance:
        if 'c' in params:
            newCov = stats['obsobs'] / stats['ntotobs'] - \
                     np.dot(GammaX.T, newMean) / stats['ntotobs']
            self._covars_ = 0.5 * (newCov + newCov.T)
        ## print newCov #DEBUG
        # should check if covars is still valid (check the det)
        
        # update transition:
        #    done in each HMM, in call of super() above.
    
    def _initialize_sufficient_statistics_var(self):
        stats = super(GaussianFHMM, \
                      self)._initialize_sufficient_statistics_var()
        stats['obsobs'] = np.zeros([self.n_features, self.n_features])
        stats['ntotobs'] = 0
        return stats
    
    def _accumulate_sufficient_statistics_var(self, stats, seq,
                                              frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """
        params: 'c' covars
                'm' means
                't' transition matrices (in self.HMM)
                's' start probabilities (note: in scikits.learn.hmm, it's only
                                         meaningful with many observed
                                         sequences)
                
        """
        
        super(GaussianFHMM, self)._accumulate_sufficient_statistics_var(\
                                              stats, seq, frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params)
        stats['obsobs'] += np.dot(seq.T, seq)
        stats['ntotobs'] += seq.shape[0]
    
    def _accumulate_sufficient_statistics_var_chain(self, stats, seq,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """
        params: 'c' covars
                'm' means
                't' transition matrices (in self.HMM)
                's' start probabilities (note: in scikits.learn.hmm, it's only
                                         meaningful with many observed
                                         sequences)
                
        """
        
        super(GaussianFHMM, self)._accumulate_sufficient_statistics_var_chain(\
                                              stats, seq, 
                                              posteriors, fwdlattice,
                                              bwdlattice, params)
        stats['obsobs'] += np.dot(seq.T, seq)
        stats['ntotobs'] += seq.shape[0]

    def _init_posterior_var_lms(self, obs, verbose=False, debug=False):
        """least mean squares solution of obs = means*posteriors
        """
        eps = 1e-6
        nframes = obs.shape[0]
        means = np.concatenate(self.means.values()).T
        # h = np.zeros([nframes, means.shape[1]])
        # for n in range(nframes):
        #    if verbose:
        #        print 'frame',n
        #    h[n] = nnls(means, obs[n])[0]
        h = linalg.lstsq(means, obs.T)[0].T
        h = np.maximum(h, eps)
        posteriors = {}
        logPost = {}
        for n in range(self.n_chains):
            n0 = np.sum(self.n_components_per_chain[:n])
            n1 = n0 + self.n_components_per_chain[n]
            if self.withNoiseF0 or self.withFlatFilter:
                #do not take last element, which is noise
                h[:, n1-1] = 0
            posteriors[n] = normalize(h[:,n0:n1], axis=1)
            logPost[n] = np.log(posteriors[n])
        
        return posteriors, logPost
        
    def _init_posterior_var_nmf(self, obs, verbose=False, debug=False):
        """posteriors as normalized nmf amplitude coefficients,
        if it makes sense (for a given observation likelihood)
        """
        
        nmfIter = 100
        nmfEps = 1e-4
        nmfMin = obs.min()
        nuObs = np.maximum(obs - nmfMin, nmfEps)
        
        nframes = obs.shape[0]
        means = np.concatenate(self.means.values())
        means = means - means.min() + nmfEps
        nmeans = means.shape[0]
        ##amplitudes = obs.max() / means.max() * \
        ##             (np.random.rand(nframes, nmeans))
        amplitudes = nuObs.max() / means.max() * \
                     np.ones((nframes, nmeans))
        amplitudes = np.maximum(amplitudes, nmfEps)
        
        for n in range(nmfIter):
            if verbose:
                print "iter",n,"over",nmfIter
            # NMF with IS rules:
            obsHat = np.dot(amplitudes, means)
            denom = np.dot(1. / np.maximum(obsHat, nmfEps), means.T)
            num = np.dot(nuObs / \
                         np.maximum(obsHat**2, nmfEps),
                         means.T)
            amplitudes = amplitudes * num / np.maximum(denom, nmfEps)
            amplitudes = np.maximum(amplitudes, nmfEps)
            if debug:
                plt.figure(200)
                plt.clf()
                plt.imshow(np.log(amplitudes), origin='lower',
                           interpolation='nearest')
                plt.colorbar()
                plt.draw()
                
        
        posteriors = {}
        logPost = {}
        
        for n in range(self._n_chains): 
            n0 = np.sum(self.n_components_per_chain[:n])
            n1 = n0 + self.n_components_per_chain[n]
            posteriors[n] = normalize(amplitudes[:,n0:n1], axis=1)
            logPost[n] = np.log(posteriors[n])
        
        return posteriors, logPost


