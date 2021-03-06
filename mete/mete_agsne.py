from __future__ import division
import numpy as np
from scipy.optimize import bisect, fsolve
import mete_distributions as medis

def get_agsne_lambdas(G, S, N, E, version = 'precise'):
    """Obtain the Lagrange multipliers lambda 1, beta (lambda 1 + lambda 2),  lambda 3 for AGSNE.
    
    Two versions are available. For version = 'appox', estimations are obtained using Table 2 in Harte et al. 2015.
    For version = 'precise', estimations are obtained using S-16 to S-19 in Harte et al. 2015.
    
    """
    lambda3 = G / (E - N)
    y1 = lambda x: x * np.log(x) - S / G # Here x is 1/lambda1
    y2 = lambda x: x * np.log(x) - N / G # here x is 1 / beta
    inv_1 = fsolve(y1, 1)[0]
    inv_beta = fsolve(y2, 1)[0]    
    if version == 'approx':
        out = [1/inv_1, 1/inv_beta, lambda3]
    else:
        # Try to provide a better initial guess by fixing lambda1 and solving for beta
        exp_1_approx = np.exp(-1/inv_1)
        Slist = np.arange(1, S + 1)
        #y3 = lambda x: np.sum(exp_1_approx ** Slist * np.log(1 / (1 - x ** Slist))) / np.sum(exp_1_approx ** Slist / Slist * np.log(1 / (1 - x ** Slist))) - S / G
        y3 = lambda x: np.sum((exp_1_approx * x) ** Slist / (1 - x ** Slist)) / np.sum(exp_1_approx ** Slist / Slist * np.log(1 / (1 - x ** Slist))) - N / G
        exp_beta_approx = bisect(y3, 10**-15, 1 - 10**-15)        
        def lambdas(x):
            exp_1, exp_beta = x # The two elements are exp(-lambda1), and exp(-beta)
            f1 = np.sum(exp_1 ** Slist * np.log(1 / (1 - exp_beta ** Slist))) / np.sum(exp_1 ** Slist / Slist * np.log(1 / (1 - exp_beta ** Slist))) - S / G 
            f2 = np.sum((exp_1 * exp_beta) ** Slist / (1 - exp_beta ** Slist)) / np.sum(exp_1 ** Slist / Slist * np.log(1 / (1 - exp_beta ** Slist))) - N / G
            return(f1, f2)
        exp_1, exp_beta = fsolve(lambdas, np.array((exp_1_approx, exp_beta_approx)), factor = 0.1, maxfev = 500)
        out = [-np.log(exp_1), -np.log(exp_beta), lambda3]
    return (out)
        
def agsne_lambda3_z(lambda1, beta, S):
    """Compute lambda3 * Z, a widely used constant."""
    Slist = np.array(range(1, S + 1))
    ans = np.sum(np.exp(-lambda1*Slist) / Slist * np.log(1 / (1 - np.exp(-beta * Slist))))
    return ans

def get_mete_agsne_rad(G, S, N, E, version='precise', pars = None):
    """Compute RAD predicted by the AGSNE.
    
    Arguments:
    G, S, N, E - state variables (number of genera, number of species, number of individuals, total metabolic rate)
    Keyword arguments:
    version - which version is used to calculate the state variables, can take 'precise' or 'approx'
    pars - a list of Langrage multipliers [lambda1, beta, lambda3, Z], if these are already available. 
        If None, Langrage multipliers are calculated from get_agsne_lambdas(). 
    
    Return a list of expected abundances with length S, ranked from high to low.
    """    
    if pars is None:
        lambda1, beta, lambda3 = get_agsne_lambdas(G, S, N, E, version = version)
        lambda3z = agsne_lambda3_z(lambda1, beta, S)
        pars = [lambda1, beta, lambda3, lambda3z / lambda3]
    
    sad = medis.sad_agsne([G, S, N, E], pars)
    rank = range(1, int(S)+1)
    abundance  = list(np.empty([S]))
    cdf_obs = [(rank[i]-0.5) / S for i in range(0, int(S))]
    i, j = 1, 0
    cdf_cum = 0 
    while i <= N + 1:
        cdf_cum += sad.pmf(i)
        while cdf_cum >= cdf_obs[j]: 
            abundance[j] = i
            j += 1
            if j == S:
                abundance.reverse()
                return abundance
        i += 1
    
def get_mete_agsne_isd(G, S, N, E, version='precise', pars = None):
    """Compute the individual size at each rank in ISD predicted by AGSNE.
    
    Arguments:
    G, S, N, E - state variables (number of genera, number of species, number of individuals, total metabolic rate)
    Keyword arguments:
    version - which version is used to calculate the state variables, can take 'precise' or 'approx'
    pars - a list of Langrage multipliers [lambda1, beta, lambda3 * Z], if these are already available. 
        If None, Langrage multipliers are calculated from get_agsne_lambdas(). 
    
    Return a list of expected abundances with length S, ranked from high to low.
    """    
    if pars is None:
        lambda1, beta, lambda3 = get_agsne_lambdas(G, S, N, E, version = version)
        lambda3z = agsne_lambda3_z(lambda1, beta, S)
        pars = [lambda1, beta, lambda3, lambda3z / lambda3]

    psi = medis.psi_agsne([G, S, N, E], pars)
    cdf_obs = [(i - 0.5) / N for i in range(1, int(N) + 1)]
    isd = np.array([psi.ppf(x) for x in cdf_obs])
    return isd
