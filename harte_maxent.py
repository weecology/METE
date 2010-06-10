# Module for fitting and testing Harte et al.'s maximum entropy models

from __future__ import division
from math import log
from scipy.optimize import fsolve

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
    assert approx == 'yes' or approx == 'no', """approx must be 'yes', 'no', or
    blank"""
    if approx == 'no':
        y = lambda x: 1/log(1/(1-x))*x/(1-x)-N/S # x = e**-lambda_1 in B.4
    else:
        y = lambda x: x*log(1/x)-S/N
    XSTART = 0.999999
    exp_neg_lambda_1 = fsolve(y, XSTART)
    lambda_1 = -1*log(exp_neg_lambda_1)
    return lambda_1