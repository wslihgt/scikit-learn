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

# Data as part of Ghahramani's Matlab implementation of his variational FHMM
# algorithm
#
# 
import scipy.io

import os

absPath = os.path.abspath('./sklearn/tests/')
print os.path.join(absPath,'X.mat')
struc = scipy.io.loadmat(os.path.join(absPath,'X.mat'))
X = struc['X']

fghmm = fhmm.GaussianFHMM(n_components_per_chain=[2,2])

## fghmm.fit_var([X, X[100:], X[50:], X[::-1]], n_iter=30, n_innerLoop=20,
##               verbose=True, params='mc')

##fghmm.means = [np.array([[-5], [5]]),
##               np.array([[-2], [2]]),
##               np.array([[-0.5], [0.5]])]

fghmm.fit_var([X], n_iter=100, n_innerLoop=30,
              verbose=True, params='stmc',
              init_params='smtc')

mpost, states, posteriors = fghmm.decode_var(X, n_innerLoop=20, verbose=True,
                                             debug=True)

import matplotlib.pyplot as plt
plt.ion()
plt.figure()
plt.plot(X,'.')
plt.plot(mpost)
for n, state in states.items():
    plt.plot(state)
