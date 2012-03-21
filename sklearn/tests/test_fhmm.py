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


class SeedRandomNumberGeneratorTestCase(TestCase):
    seed = 9

    def __init__(self, *args, **kwargs):
        self.setUp()
        TestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        self.prng = np.random.RandomState(self.seed)


class TestBaseHMM(SeedRandomNumberGeneratorTestCase):

    class StubHMM(fhmm._BaseFactHMM):

        def _compute_log_likelihood(self, X):
            return self.framelogprob

        def _generate_sample_from_state(self):
            pass

        def _init(self):
            pass

    def setup_example_hmm(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        h = self.StubHMM(n_states=[2, 1])
        h.HMM[0].transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.HMM[0].startprob_ = [0.5, 0.5]
        h.HMM[1].transmat_ = [[1.0]]
        h.HMM[1].startprob_ = [1.0]
        framelogprob = np.log([[0.9, 0.2],
                               [0.9, 0.2],
                               [0.1, 0.8],
                               [0.9, 0.2],
                               [0.9, 0.2]])
        # Add dummy observations to stub.
        h.framelogprob = framelogprob
        return h, framelogprob

    def test_init(self):
        h, framelogprob = self.setup_example_hmm()
        for params in [('transmat_',), ('startprob_', 'transmat_')]:
            d = dict((x[:-1], getattr(h, x)) for x in params)
            h2 = self.StubHMM(n_states=h.n_states, **d)
            self.assertEqual(h.n_components, h2.n_components)
            self.assertEqual(h.n_states, h2.n_states)
            for p in params:
                assert_array_almost_equal(getattr(h, p), getattr(h2, p))

    def test_do_forward_pass(self):
        """
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
        h, framelogprob = self.setup_example_hmm()
        
        chain = 0
        logprob, fwdlattice = h._do_forward_pass_var_chain(framelogprob,
                                                           chain=chain)
        
        reflogprob = -3.3725
        print reflogprob, logprob
        self.assertAlmostEqual(logprob, reflogprob, places=4)
        
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_array_almost_equal(np.exp(fwdlattice), reffwdlattice, 4)
        
        chain = 1
        logprob, fwdlattice = h._do_forward_pass_var_chain(framelogprob,
                                                           chain=chain)
        print reflogprob, logprob
        reffwdlattice = np.array([[1.0],
                                  [1.0],
                                  [1.0],
                                  [1.0],
                                  [1.0]])
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
