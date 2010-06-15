# Module for fitting and testing Harte et al.'s maximum entropy models

# To Do:
#    See if applying A.5 to B.1 yeilds an assumption free soln to lambda_1
#    get_lambda_P does not find a root because it appears that there is no root
#       for equation B.5 - investigate

from __future__ import division
from math import log
from scipy.optimize import bisect

# Set the distance from the undefined boundaries of the Lagrangian multipliers
# to set the upper and lower boundaries for the numerical root finders
DIST_FROM_BOUND = 10 ** -10

def get_lambda_1(S, N, approx='no'):
    """Solve for lambda_1 from Harte et al. 2008
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    approx -- 'no' uses eq. B.4, which uses minimal approximations
              'yes' uses eq. 7b which uses an additional approximation
              the default is 'no' and using the default is strongly recommended
              unless there is a very clear reason to do otherwise.
              
    """
    assert type(S) is int, "S must be an integer"
    assert type(N) is int, "N must be an integer"
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    assert approx == 'yes' or approx == 'no', """approx Bst be 'yes', 'no', or
    blank"""
    
    BOUNDS = [0, 1]
    
    # Solve for lambda_1 using the substitution x = e**-lambda_1
    if approx == 'no':
        y = lambda x: 1 / log(1 / (1 - x)) * x / (1 - x) - N / S
    else:
        y = lambda x: x * log(1 / x) - S / N
    exp_neg_lambda_1 = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, BOUNDS[1] -
                              DIST_FROM_BOUND)
    lambda_1 = -1 * log(exp_neg_lambda_1)
    return lambda_1

def get_lambda_P(A, A_0, n_0):
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
        lambda_P = 0
    else:
        # Solve for lambda_P using the substitution x = e**-lambda_P
        BOUNDS = [0, 1]
        y = lambda x: 1 / (1 - x ** (n_0 + 1)) * (x / (1 - x) - x ** (n_0 +1) *
                                                  (n_0 + 1 / (1 - x))) - (n_0 * A /
                                                                          A_0)
        exp_neg_lambda_P = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, BOUNDS[1] -
                                  DIST_FROM_BOUND)
        lambda_P = -1 * log(exp_neg_lambda_P)
    return lambda_P