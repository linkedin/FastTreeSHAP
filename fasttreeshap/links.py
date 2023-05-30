import numpy as np
from numba import njit

@njit
def identity(x):
    """ A no-op link function.
    """
    return x
@njit
def _identity_inverse(x):
    return x
identity.inverse = _identity_inverse

@njit
def logit(x):
    """ A logit link function useful for going from probability units to log-odds units.
    """
    return np.log(x/(1-x))
@njit
def _logit_inverse(x):
    return 1/(1+np.exp(-x))
logit.inverse = _logit_inverse
