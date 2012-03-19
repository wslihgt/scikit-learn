import scipy.io
from sklearn import fhmm
reload(fhmm)

struc = scipy.io.loadmat('X.mat')
X = struc['X']

fghmm = fhmm.GaussianFHMM(n_states=[2,2])

## fghmm.fit_var([X, X[100:], X[50:], X[::-1]], n_iter=30, n_innerLoop=20,
##               verbose=True, params='mc')

fghmm.fit_var([X], n_iter=100, n_innerLoop=30,
              verbose=True, params='stmc')

mpost, states, posteriors = fghmm.decode_var(X, n_innerLoop=20, verbose=True,
                                             debug=True)

import matplotlib.pyplot as plt
plt.ion()
plt.figure()
plt.plot(X,'.')
plt.plot(mpost)
for n, state in states.items():
    plt.plot(state)
