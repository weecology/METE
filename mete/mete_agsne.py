from __future__ import division
import numpy as np
from scipy.optimize import bisect, fsolve, fmin_l_bfgs_b
# G, S, N, E = [10, 20, 200, 1000]
def get_lambdas(G, S, N, E, version = 'precise'):
    """Obtain the Lagrange multipliers lambda 1, beta (lambda 1 + lambda 2),  lambda 3 for ASGNE.
    
    Two versions are available. For version = 'appox', estimations are obtained using Table 2 in Harte et al. 2015.
    For version = 'precise', estimations are obtained using S-16 to S-19 in Harte et al. 2015.
    
    """
    lambda3 = G / (E - N)
    if verion == 'approx':
        y1 = lambda x: x * np.log(x) - S / G # Here x is 1/lambda1
        y2 = lambda x: x * np.log(x) - N / G # here x is 1 / beta
        inv_1 = fsolve(y1, 1)[0]
        inv_beta = fsolve(y2, 1)[0]
        return([1/inv_1, 1/inv_beta, lambda3])
    else:
        def lambdas(x):
            exp_1, exp_beta = x # The two elements are exp(-lambda1), and exp(-beta)
            Slist = range(1, S + 1)
            f1 = np.sum(exp_1 ** Slist * np.log(1 / (1 - exp_beta ** Slist))) / np.sum(exp_1 ** Slist / Slist * np.log(1 / (1 - exp_beta ** Slist))) - S / G 
            f2 = np.sum((exp_1 * exp_beta) ** Slist / (1 - exp_beta ** Slist)) / np.sum(exp_1 ** Slist / Slist * np.log(1 / (1 - exp_beta ** Slist))) - N / G
            return(f1, f2)
        exp_1, exp_beta = fsolve(lambdas, np.array((np.exp(-1 / inv_1), np.exp(-1 / inv_beta))), factor = 0.1, maxfev = 500)
        return ([-np.log(exp_1), -np.log(exp_beta), lambda3])
        
