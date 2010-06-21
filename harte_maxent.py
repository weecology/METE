# Module for fitting and testing Harte et al.'s maximum entropy models

# To Do:
#    See if applying A.5 to B.1 yeilds an assumption free soln to lambda_1
#    get_lambda_P does not find a root because it appears that there is no root
#       for equation B.5 - investigate

from __future__ import division
from math import log, exp
from scipy.optimize import bisect

# Set the distance from the undefined boundaries of the Lagrangian multipliers
# to set the upper and lower boundaries for the numerical root finders
DIST_FROM_BOUND = 10 ** -10

def get_lambda_sad(S, N, approx='no'):
    """Solve for lambda_1 from Harte et al. 2008
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    
    """
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    BOUNDS = [0, 1]
    
    # Solve for lambda_sad using the substitution x = e**-lambda_1
    y = lambda x: 1 / log(1 / (1 - x)) * x / (1 - x) - N / S
    exp_neg_lambda_sad = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, BOUNDS[1] -
                              DIST_FROM_BOUND)
    lambda_sad = -1 * log(exp_neg_lambda_sad)
    return lambda_sad

def get_lambda_spatialdistrib(A, A_0, n_0):
    """Solve for lambda_P from Harte et al. 2008
    
    Keyword arguments:
    A -- the spatial scale of interest
    A_0 -- the maximum spatial scale under consideration
    n_0 -- the number of individuals of the focal species at scale A_0
    
    """
    assert type(n_0) is int, "n must be an integer"
    assert A > 0 and A_0 > 0, "A and A_0 must be greater than 0"
    assert A <= A_0, "A must be less than or equal to A_0"
    
    if A == A_0 / 2:
        # Special case where A = A_0/2 from Harte et al. 2009 after eq. 6
        lambda_spatialdistrib = 0
    else:
        # Solve for lambda_P using the substitution x = e**-lambda_P
        BOUNDS = [0, 1]
        y = lambda x: 1 / (1 - x ** (n_0 + 1)) * (x / (1 - x) - x ** (n_0 +1) *
                                                  (n_0 + 1 / (1 - x))) - (n_0 * A /
                                                                          A_0)
        exp_neg_lambda_spatialdistrib = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, BOUNDS[1] -
                                  DIST_FROM_BOUND)
        lambda_spatialdistrib = -1 * log(exp_neg_lambda_spatialdistrib)
    return lambda_spatialdistrib

def downscale_sar(A, S, N, Amin):
    """Predictions for downscaled SAR using Eq. 7 from Harte et al. 2009"""
    lambda_sad = get_lambda_sad(S, N)
    x = exp(-lambda_sad)
    S = S / x - N * (1 - x) / (x - x ** (N + 1)) * (1 - x ** N / (N + 1))
    A /= 2
    N /= 2
    if A <= Amin:
        return ([A], [S])
    else:
        down_scaled_data = downscale_sar(A, S, N, Amin)
        return (down_scaled_data[0] + [A], down_scaled_data[1] + [S])

def upscale_sar(A, S, N, Amax):
    """Predictions for downscaled SAR using Eq. 7 from Harte et al. 2009"""

def sar(A_0, S_0, N_0, Amin, Amax):
    """Harte et al. 2009 predictions for the species area relationship
    
    Takes a minimum and a maximum area along with the area, richness, and
    abundance at some anchor scale and determines the richness at all bisected
    and/or doubled scales so as to include Amin and Amax.
    
    """
    # This is where we will deal with adding the anchor scale to the results