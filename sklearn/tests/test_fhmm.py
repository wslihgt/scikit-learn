"""Test script for the FHMM module for sklearn

Based on the HMM test script: sklearn/tests/test_hmm.py

Jean-Louis Durrieu, 2012
"""

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn import fhmm
from sklearn.utils.extmath import logsumexp

rng = np.random.RandomState(0)
np.seterr(all='warn')

def test_normalize_zeros():
    """test_normalize_zeros
    Checks whether the normalize function in fhmm indeed does _not_
    normalize when the column is 0.
    
    question: is this a desired behavior?
    """
    A = np.zeros(2)
    for axis in range(1):
        Anorm = fhmm.normalize(A, axis)
        assert np.all(np.allclose(Anorm.sum(axis), 0.0))

def test_normalize_1D():
    A = rng.rand(2) + 1.0
    for axis in range(1):
        Anorm = fhmm.normalize(A, axis)
        assert np.all(np.allclose(Anorm.sum(axis), 1.0))

def test_normalize_3D():
    A = rng.rand(2, 2, 2) + 1.0
    for axis in range(3):
        Anorm = fhmm.normalize(A, axis)
        assert np.all(np.allclose(Anorm.sum(axis), 1.0))


class SeedRandomNumberGeneratorTestCase(TestCase):
    seed = 9

    def __init__(self, *args, **kwargs):
        self.setUp()
        TestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        self.prng = np.random.RandomState(self.seed)


class TestBaseHMM(SeedRandomNumberGeneratorTestCase):

    class StubHMM(fhmm._BaseFactHMM):

        def _compute_var_params(self, obs, **kwargs):
            return self.framelogprob

        def _compute_var_params_chain(self, obs, posteriors, chain,
                                      **kwargs):
            return self.framelogprob[chain]

        def _compute_log_likelihood(self, obs):
            return self.framelogprob

        def _generate_sample_from_state(self):
            pass
        
        def _init(self):
            pass
        
    def setup_example_hmm_none(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        h = self.StubHMM(n_components_per_chain=[2],
                         startprob_per_chain=None,
                         transmat_per_chain=None)
        #h.HMM[0].transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        #h.HMM[0].startprob_ = [0.5, 0.5]
        #h.HMM[1].transmat_ = [[1.0]]
        #h.HMM[1].startprob_ = [1.0]
        framelogprob = np.log([[0.9, 0.2],
                               [0.9, 0.2],
                               [0.1, 0.8],
                               [0.9, 0.2],
                               [0.9, 0.2]])
        # Add dummy observations to stub.
        h.framelogprob = framelogprob
        return h, framelogprob
        
    def setup_example_hmm(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        h = self.StubHMM(n_components_per_chain=[2],
                         startprob_per_chain=[[0.5, 0.5]],
                         transmat_per_chain=[[[0.7, 0.3], [0.3, 0.7]]])
        #h.HMM[0].transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        #h.HMM[0].startprob_ = [0.5, 0.5]
        #h.HMM[1].transmat_ = [[1.0]]
        #h.HMM[1].startprob_ = [1.0]
        framelogprob = np.log([[0.9, 0.2],
                               [0.9, 0.2],
                               [0.1, 0.8],
                               [0.9, 0.2],
                               [0.9, 0.2]])
        # Add dummy observations to stub.
        h.framelogprob = framelogprob
        return h, framelogprob
    
    def setup_example_fhmm(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        h = self.StubHMM(n_components_per_chain=[2, 2],
                         startprob_per_chain=[[0.5, 0.5],
                                              [0.5, 0.5]],
                         transmat_per_chain=[[[0.7, 0.3], [0.3, 0.7]],
                                             [[0.7, 0.3], [0.3, 0.7]]])
        #h.HMM[0].transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        #h.HMM[0].startprob_ = [0.5, 0.5]
        #h.HMM[1].transmat_ = [[1.0]]
        #h.HMM[1].startprob_ = [1.0]
        framelogprob = [np.log([[0.9, 0.2],
                                [0.9, 0.2],
                                [0.1, 0.8],
                                [0.9, 0.2],
                                [0.9, 0.2]]),
                        np.log([[0.9, 0.2],
                                [0.9, 0.2],
                                [0.1, 0.8],
                                [0.9, 0.2],
                                [0.9, 0.2]])]
        # Add dummy observations to stub.
        h.framelogprob = framelogprob
        return h, framelogprob
    
    def test_init_none(self):
        h, framelogprob = self.setup_example_hmm_none()
        for params in [('transmat_per_chain_',),
                       ('startprob_per_chain_',
                        'transmat_per_chain_')]:
            d = dict((x[:-1], None) for x in params)
            h2 = self.StubHMM(n_components_per_chain=\
                              h.n_components_per_chain,
                              **d)
            self.assertEqual(h.n_components, h2.n_components)
            self.assertEqual(h.n_components_per_chain,
                             h2.n_components_per_chain)
            for p in params:
                assert_array_almost_equal(getattr(h, p), getattr(h2, p))
    
    def test_init(self):
        h, framelogprob = self.setup_example_hmm()
        for params in [('transmat_per_chain_',),
                       ('startprob_per_chain_',
                        'transmat_per_chain_')]:
            d = dict((x[:-1], getattr(h, x)) for x in params)
            h2 = self.StubHMM(n_components_per_chain=\
                              h.n_components_per_chain,
                              **d)
            self.assertEqual(h.n_components, h2.n_components)
            self.assertEqual(h.n_components_per_chain,
                             h2.n_components_per_chain)
            for p in params:
                assert_array_almost_equal(getattr(h, p), getattr(h2, p))
    
    def test_init_fhmm(self):
        h, framelogprob = self.setup_example_fhmm()
        for params in [('transmat_per_chain_',),
                       ('startprob_per_chain_',
                        'transmat_per_chain_')]:
            d = dict((x[:-1], getattr(h, x)) for x in params)
            h2 = self.StubHMM(n_components_per_chain=\
                              h.n_components_per_chain,
                              **d)
            self.assertEqual(h.n_components, h2.n_components)
            self.assertEqual(h.n_components_per_chain,
                             h2.n_components_per_chain)
            for p in params:
                assert_array_almost_equal(getattr(h, p), getattr(h2, p))
                
    def test_do_forward_pass(self):
        """test_do_forward_pass
        This actually calls a regular HMM method.
        
        """
        h, framelogprob = self.setup_example_hmm()

        logprob, fwdlattice = h._do_forward_pass(framelogprob)

        reflogprob = -3.3725
        self.assertAlmostEqual(logprob, reflogprob, places=4)
        
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_array_almost_equal(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_forward_pass_var(self):
        """
        
        """
        h, framelogprob = self.setup_example_fhmm()
        
        chain = 0
        logprob, fwdlattice = h._do_forward_pass_var_chain(framelogprob[chain],
                                                           chain=chain)
        
        reflogprob = -3.3725
        
        self.assertAlmostEqual(logprob, reflogprob, places=4)
        
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_array_almost_equal(np.exp(fwdlattice), reffwdlattice, 4)
        
        chain = 1
        logprob, fwdlattice = h._do_forward_pass_var_chain(framelogprob[chain],
                                                           chain=chain)
        
        self.assertAlmostEqual(logprob, reflogprob, places=4)
        
        assert_array_almost_equal(np.exp(fwdlattice), reffwdlattice, 4)
    
    def test_do_backward_pass(self):
        """
        This actually calls a regular HMM method.
        
        """
        h, framelogprob = self.setup_example_hmm()

        bwdlattice = h._do_backward_pass(framelogprob)

        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        assert_array_almost_equal(np.exp(bwdlattice), refbwdlattice, 4)

    
    def test_do_backward_pass_var(self):
        """
        This actually calls a regular HMM method.
        
        """
        h, framelogprob = self.setup_example_fhmm()
        
        chain = 0
        
        bwdlattice = h._do_backward_pass_var_chain(framelogprob[chain],
                                                   chain=chain)
        
        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        
        assert_array_almost_equal(np.exp(bwdlattice), refbwdlattice, 4)
        
        chain = 1
        
        bwdlattice = h._do_backward_pass_var_chain(framelogprob[chain],
                                                   chain=chain)
        
        assert_array_almost_equal(np.exp(bwdlattice), refbwdlattice, 4)
    
    def test_do_viterbi_pass(self):
        h, framelogprob = self.setup_example_hmm()
        
        logprob, state_sequence = h._do_viterbi_pass(framelogprob)
        
        refstate_sequence = [0, 0, 1, 0, 0]
        assert_array_equal(state_sequence, refstate_sequence)
        
        reflogprob = -4.4590
        self.assertAlmostEqual(logprob, reflogprob, places=4)

    def test_eval(self):
        h, framelogprob = self.setup_example_hmm()
        nobs = len(framelogprob)
        
        logprob, posteriors = h.eval([], estimation_method='full')
        
        assert_array_almost_equal(posteriors.sum(axis=1), np.ones(nobs))
        
        reflogprob = -3.3725
        self.assertAlmostEqual(logprob, reflogprob, places=4)
        
        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        assert_array_almost_equal(posteriors, refposteriors, decimal=4)

prng = np.random.RandomState(10)

class GaussianHMMParams(object):
    """
    Example from http://mlg.eng.cam.ac.uk/zoubin/software/fhmm.tar.gz
    """
    n_components_per_chain = [2, 2]
    n_components = 4
    n_chains = 2
    n_features = 1
    startprob_per_chain = [prng.rand(n_components_per_chain[n]) \
                           for n in range(n_chains)]
    for n in range(n_chains):
        startprob_per_chain[n] = startprob_per_chain[n] / \
                                 startprob_per_chain[n].sum()
    
    transmat_per_chain = [prng.rand(n_components_per_chain[n],
                                    n_components_per_chain[n]) \
                          for n in range(n_chains)]
    for n in range(n_chains):
        transmat_per_chain[n] /= \
            np.tile(transmat_per_chain[n].sum(axis=1)[:, np.newaxis],
                    (1, n_components_per_chain[n]))
    
    means_ = [prng.randint(-20, 20, (n_components_per_chain[n], n_features))\
              for n in range(n_chains)]
    # Gaussian FHMM with variational only works with tied covar:
    covars_ = {'spherical': (1.0 + 2 * np.dot(prng.rand(n_components, 1),
                                        np.ones((1, n_features)))) ** 2,
              'tied': (make_spd_matrix(n_features, random_state=0)
                       + np.eye(n_features)),
              'diag': (1.0 + 2 * prng.rand(1, n_features)) ** 2,
              'full': np.array(
            [make_spd_matrix(n_features,
                             random_state=0) + np.eye(n_features)
             for x in xrange(n_components)])}
    expanded_covars = {'spherical': [np.eye(n_features) * cov
                                     for cov in covars_['spherical']],
                       'diag': [np.diag(cov) for cov in covars_['diag']],
                       'tied': [covars_['tied']] * n_components,
                       'full': covars_['full']}

toyData = np.vstack(\
      [ 2.61649535,  2.56268391,  6.50750802,  7.53516069,  6.43034875,
        2.66961425,  2.50590598,  2.67970718,  2.52640685,  7.58716733,
        6.35538285,  2.42988347,  2.62459821,  2.4361023 ,  2.55773502,
        3.46399704,  7.48644237,  6.36506615,  7.37295501,  2.59845703,
        2.49551194,  2.42010555,  2.42348276,  2.58617349,  2.49437749,
        2.55134782, -6.46033191, -6.4243781 , -7.4599514 , -7.63413807,
       -7.4624959 , -3.38748382, -2.42713584, -3.73774543, -3.52737824,
       -7.53229399, -7.46820121, -6.55111722, -7.50020413, -6.3393489 ,
       -6.41523514, -6.47318992,  2.40765109,  2.49295006,  2.51478914,
        3.44429064,  2.46632943,  3.54152275,  2.65578135,  2.25557011,
        2.39018046,  2.61226479,  6.55816673,  2.47286457,  2.54141913,
        6.40221858,  7.39785338,  6.5317688 ,  6.65161078,  7.57494325,
        7.44922996,  6.58852994,  7.47519064,  7.4273751 ,  6.45549597,
        6.43870889,  7.47908559,  7.55621478,  7.39360771,  7.53515889,
        7.61329999,  6.51499942,  6.57031441,  2.49475884,  2.70184961,
        2.59241594,  2.31858853,  2.50349733,  3.31921379,  2.60281925,
        2.53946003,  2.56394056,  3.58742129,  2.67524017,  2.46799492,
       -6.51374138, -2.43842304, -2.40221059, -7.61153477, -2.55500214,
       -2.49601151, -2.74828425, -7.38413453, -7.60262795, -6.3846513 ,
        7.42135434,  7.56348086,  6.58204098,  7.48239735,  7.55624739,
        6.48725571,  6.55541716,  6.39026557,  6.42686986,  7.64047319,
        6.43797858,  7.52371488, -2.6586847 , -6.54014848, -6.57706923,
       -6.52626805, -7.40235105, -7.4022185 , -6.38299789, -2.48406891,
       -2.45004791, -2.60553751,  7.45492568,  6.62703782,  7.58986936,
        2.54387051,  2.37526557,  2.53246669,  7.53900704,  6.45948617,
        7.52923149,  7.75659102,  7.45421844,  6.3389173 ,  6.23304762,
        6.42403034,  7.43252791,  7.38283128,  2.703293  ,  2.5968481 ,
        7.5670292 ,  2.5420146 ,  7.21272487,  7.66858741,  7.50279246,
        7.40979694,  6.29467425,  6.50890863,  7.70870991,  7.53651185,
        7.58461055,  7.48154623,  6.60307144,  6.34723773,  7.5964939 ,
        6.55261625,  7.48155459,  7.51987828, -3.34095732, -3.49678084,
        7.58891637,  2.37008475,  2.61825731,  7.68174717,  7.44156979,
        2.39893262,  6.40395017,  7.56911596,  7.42413818,  7.49030283,
        2.3593051 ,  3.60308125,  2.42401256,  2.58741272,  2.5761127 ,
       -7.51659235, -3.46990926, -3.53224673, -7.53684113, -3.38521047,
       -2.49585697, -2.60980497, -2.34332763, -3.60484234, -6.45772763,
       -6.58444144,  6.46883702,  7.53978105,  6.6049786 ,  7.46592044,
        2.5336297 ,  2.47786392,  2.50166495,  2.38076388, -6.51316463,
       -7.35124757,  2.41631788,  2.36990181,  2.65741319,  3.616604  ,
        2.57864297,  2.35383612,  2.65544659,  2.44024646,  2.37894321])

def test_ghahramani_example():
    """test_ghahramani_example
    
    Test provided in the original MATLAB package by Ghahramani:
    http://mlg.eng.cam.ac.uk/zoubin/software/fhmm.tar.gz
    
    """
    X = toyData
    
    fghmm = fhmm.GaussianFHMM(n_components_per_chain=[2,2,2])
    
    fghmm.means = [np.array([[-5], [5]]),
                   np.array([[-2], [2]]),
                   np.array([[-0.5], [0.5]])]
    
    fghmm.fit_var([X], n_iter=100, n_innerLoop=30,
                  verbose=True, params='stmc',
                  init_params='stc')
    
    mpost, states, posteriors = fghmm.decode_var(X, n_innerLoop=20,
                                                 verbose=True,
                                                 debug=True)
    
    mpostTrue = np.vstack(\
        [ 2.66641872,  2.68037987,  7.05350948,  7.05350948,  7.03954833,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  7.05350948,
          7.03954833,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          2.68037987,  7.05350948,  7.03954833,  7.05350948,  2.68037987,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          2.68037987, -7.0992669 , -7.0992669 , -7.0992669 , -7.11322805,
          -7.11322805, -2.74009844, -2.7261373 , -2.74009844, -2.74009844,
          -7.11322805, -7.0992669 , -7.0992669 , -7.0992669 , -7.0992669 ,
          -7.0992669 , -7.0992669 ,  2.68037987,  2.68037987,  2.68037987,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          2.68037987,  2.68037987,  7.05350948,  2.68037987,  2.68037987,
          7.03954833,  7.05350948,  7.05350948,  7.05350948,  7.05350948,
          7.05350948,  7.05350948,  7.05350948,  7.05350948,  7.03954833,
          7.03954833,  7.05350948,  7.05350948,  7.05350948,  7.05350948,
          7.05350948,  7.03954833,  7.03954833,  2.68037987,  2.68037987,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          -7.0992669 , -2.7261373 , -2.7261373 , -7.0992669 , -2.7261373 ,
          -2.7261373 , -2.7261373 , -7.0992669 , -7.0992669 , -7.0992669 ,
          7.05350948,  7.05350948,  7.05350948,  7.05350948,  7.05350948,
          7.03954833,  7.03954833,  7.03954833,  7.03954833,  7.05350948,
          7.03954833,  7.05350948, -2.7261373 , -7.0992669 , -7.0992669 ,
          -7.0992669 , -7.0992669 , -7.0992669 , -7.0992669 , -2.7261373 ,
          -2.7261373 , -2.7261373 ,  7.05350948,  7.05350948,  7.05350948,
          2.68037987,  2.68037987,  2.68037987,  7.05350948,  7.03954833,
          7.05350948,  7.05350948,  7.05350948,  7.03954833,  7.03954833,
          7.03954833,  7.05350948,  7.05350948,  2.68037987,  2.68037987,
          7.05350948,  2.68037987,  7.05350948,  7.05350948,  7.05350948,
          7.05350948,  7.03954833,  7.03954833,  7.05350948,  7.05350948,
          7.05350948,  7.05350948,  7.03954833,  7.03954833,  7.05350948,
          7.05350948,  7.05350948,  7.05350948, -2.74009844, -2.74009844,
          7.05350948,  2.68037987,  2.68037987,  7.05350948,  7.05350948,
          2.68037987,  7.03954833,  7.05350948,  7.05350948,  7.05350948,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          -7.11322805, -2.74009844, -2.74009844, -7.11322805, -2.74009844,
          -2.7261373 , -2.7261373 , -2.7261373 , -2.74009844, -7.0992669 ,
          -7.0992669 ,  7.03954833,  7.05350948,  7.05350948,  7.05350948,
          2.68037987,  2.68037987,  2.68037987,  2.68037987, -7.0992669 ,
          -7.0992669 ,  2.68037987,  2.68037987,  2.68037987,  2.68037987,
          2.68037987,  2.68037987,  2.68037987,  2.68037987,  2.68037987])
    assert_array_almost_equal(mpostTrue, mpost)

def test_ghahramani_example_2chains():
    """test_ghahramani_example_2chains
    
    Test provided in the original MATLAB package by Ghahramani:
    http://mlg.eng.cam.ac.uk/zoubin/software/fhmm.tar.gz
    
    """
    X = toyData
    
    fghmm = fhmm.GaussianFHMM(n_components_per_chain=[2,2])
    
    fghmm.fit([X], n_iter=100, n_innerLoop=30,
              verbose=True, params='stmc',
              init_params='stmc')
    
    mpost, states, posteriors = fghmm.decode_var(X, n_innerLoop=20,
                                                 verbose=True,
                                                 debug=True)
    
    mpostTrue = np.vstack(\
      [ 2.67345097,  2.67345097,  7.04660672,  7.04660672,  7.04660672,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  7.04660672,
        7.04660672,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
        2.67345097,  7.04660672,  7.04660672,  7.04660672,  2.67345097,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
        2.67345097, -7.10617688, -7.10617688, -7.10617688, -7.10617688,
       -7.10617688, -2.73302113, -2.73302113, -2.73302113, -2.73302113,
       -7.10617688, -7.10617688, -7.10617688, -7.10617688, -7.10617688,
       -7.10617688, -7.10617688,  2.67345097,  2.67345097,  2.67345097,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
        2.67345097,  2.67345097,  7.04660672,  2.67345097,  2.67345097,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  2.67345097,  2.67345097,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
       -7.10617688, -2.73302113, -2.73302113, -7.10617688, -2.73302113,
       -2.73302113, -2.73302113, -7.10617688, -7.10617688, -7.10617688,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672, -2.73302113, -7.10617688, -7.10617688,
       -7.10617688, -7.10617688, -7.10617688, -7.10617688, -2.73302113,
       -2.73302113, -2.73302113,  7.04660672,  7.04660672,  7.04660672,
        2.67345097,  2.67345097,  2.67345097,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  2.67345097,  2.67345097,
        7.04660672,  2.67345097,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        7.04660672,  7.04660672,  7.04660672, -2.73302113, -2.73302113,
        7.04660672,  2.67345097,  2.67345097,  7.04660672,  7.04660672,
        2.67345097,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
       -7.10617688, -2.73302113, -2.73302113, -7.10617688, -2.73302113,
       -2.73302113, -2.73302113, -2.73302113, -2.73302113, -7.10617688,
       -7.10617688,  7.04660672,  7.04660672,  7.04660672,  7.04660672,
        2.67345097,  2.67345097,  2.67345097,  2.67345097, -7.10617688,
       -7.10617688,  2.67345097,  2.67345097,  2.67345097,  2.67345097,
        2.67345097,  2.67345097,  2.67345097,  2.67345097,  2.67345097])
    assert_array_almost_equal(mpostTrue, mpost)
    
    # MATLAB result from one "successful" run from original Ghahramani
    # code
    mpostMatlab = np.vstack([\
        2.673451, 2.673451, 7.046607, 7.046607, 7.046607, 2.673451,
        2.673451, 2.673451, 2.673451, 7.046607, 7.046607, 2.673451,
        2.673451, 2.673451, 2.673451, 2.673451, 7.046607, 7.046607,
        7.046607, 2.673451, 2.673451, 2.673451, 2.673451, 2.673451,
        2.673451, 2.673451, -7.106177, -7.106177, -7.106177, -7.106177,
        -7.106177, -2.733021, -2.733021, -2.733021, -2.733021, -7.106177,
        -7.106177, -7.106177, -7.106177, -7.106177, -7.106177, -7.106177,
        2.673451, 2.673451, 2.673451, 2.673451, 2.673451, 2.673451,
        2.673451, 2.673451, 2.673451, 2.673451, 7.046607, 2.673451,
        2.673451, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 2.673451, 2.673451, 2.673451, 2.673451, 2.673451,
        2.673451, 2.673451, 2.673451, 2.673451, 2.673451, 2.673451,
        2.673451, -7.106177, -2.733021, -2.733021, -7.106177, -2.733021,
        -2.733021, -2.733021, -7.106177, -7.106177, -7.106177, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, -2.733021,
        -7.106177, -7.106177, -7.106177, -7.106177, -7.106177, -7.106177,
        -2.733021, -2.733021, -2.733021, 7.046607, 7.046607, 7.046607,
        2.673451, 2.673451, 2.673451, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 2.673451, 2.673451, 7.046607, 2.673451, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, 7.046607, 7.046607, 7.046607,
        7.046607, 7.046607, 7.046607, -2.733021, -2.733021, 7.046607,
        2.673451, 2.673451, 7.046607, 7.046607, 2.673451, 7.046607,
        7.046607, 7.046607, 7.046607, 2.673451, 2.673451, 2.673451,
        2.673451, 2.673451, -7.106177, -2.733021, -2.733021, -7.106177,
        -2.733021, -2.733021, -2.733021, -2.733021, -2.733021, -7.106177,
        -7.106177, 7.046607, 7.046607, 7.046607, 7.046607, 2.673451,
        2.673451, 2.673451, 2.673451, -7.106177, -7.106177, 2.673451,
        2.673451, 2.673451, 2.673451, 2.673451, 2.673451, 2.673451,
        2.673451, 2.673451])
    assert_array_almost_equal(mpostMatlab, mpost)

