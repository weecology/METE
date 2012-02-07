"""Distributions for use with METE"""

from __future__ import division
import numpy as np
from scipy.stats import logser, geom, rv_discrete, rv_continuous
from scipy.optimize import bisect
from math import exp
from mete import *

class trunc_logser_gen(rv_discrete):
    """Upper truncated logseries distribution
    
    Scipy based distribution class for the truncated logseries pmf, cdf and rvs
    
    Usage:
    PMF: trunc_logser.pmf(list_of_xvals, p, upper_bound)
    CDF: trunc_logser.cdf(list_of_xvals, p, upper_bound)
    Random Numbers: trunc_logser.rvs(p, upper_bound, size=1)
    
    """
    
    def _pmf(self, x, p, upper_bound):
        if any(p < 1):
            return logser.pmf(x, p) / logser.cdf(upper_bound, p)
        else:
            x = np.array(x)
            ivals = np.arange(1, upper_bound + 1)
            normalization = sum(p ** ivals / ivals)
            pmf = (p ** x / x) / normalization
            return pmf
        
    def _cdf(self, x, p, upper_bound):
        if any(p < 1):
            return logser.cdf(x, p) / logser.cdf(upper_bound, p)
        else:
            x_list = range(1, int(x) + 1)
            cdf = sum(trunc_logser_pmf(x_list, p, upper_bound))
            return cdf
        
    def _rvs(self, p, upper_bound):    
        rvs = logser.rvs(p, size=self._size)
        for i in range(0, self._size):
            while(rvs[i] > upper_bound):
                rvs[i] = logser.rvs(p, size=1)
        return rvs

trunc_logser = trunc_logser_gen(a=1, name='trunc_logser',
                                longname='Upper truncated logseries',
                                shapes="upper_bound",
                                extradoc="""Truncated logseries
                                
                                Upper truncated logseries distribution
                                """
                                )

class psi_epsilon_gen(rv_continuous):
    """Inidividual-energy distribution predicted by METE (modified from equation 7.24)
    
    lower truncated at 1 and upper truncated at E0.
    
    Usage:
    PDF: psi_epsilon.pdf(list_of_epsilons, S0, N0, E0)
    CDF: psi_epsilon.cdf(list_of_epsilons, S0, N0, E0)
    Random Numbers: psi_epsilon.rvs(S0, N0, E0, size = 1)
    
    """        
    def _pdf(self, x, S0, N0, E0):
        beta = get_beta(S0, N0)
        lambda2 = get_lambda2(S0, N0, E0)
        sigma = beta + (E0 - 1) * lambda2
        norm_factor = lambda2 / ((np.exp(-beta) - np.exp(-beta * (N0 + 1))) / (1 - np.exp(-beta)) - 
                                (np.exp(-sigma) - np.exp(-sigma * (N0 + 1))) / (1 - np.exp(-sigma)))
        x = np.array(x)
        exp_neg_gamma = np.exp(-(beta + (x - 1) * lambda2))
        return norm_factor * exp_neg_gamma * (1 - (N0 + 1) * exp_neg_gamma ** N0 +
                                              N0 * exp_neg_gamma ** (N0 + 1)) / (1 - exp_neg_gamma) ** 2
        #Below is the exact form of equation 7.24, which seems to contain an error: 
        #return norm_factor * (exp_neg_gamma / (1 - exp_neg_gamma) ** 2 - 
                              #exp_neg_gamma ** N0 / (1 - exp_neg_gamma) *
                              #(N0 + exp_neg_gamma / (1 - exp_neg_gamma)))

    def _ppf(self, q, S0, N0, E0):
        x = []
        for q_i in q: 
            y_i = lambda t: self._cdf(t, S0, N0, E0) - q_i
            x.append(bisect(y_i, self.a, self.b, xtol = 1.490116e-08))
        return np.array(x)
    
    def _argcheck(self, *args):
        self.a = 1
        self.b = args[2]
        cond = (args[0] > 0) & (args[1] > 0) & (args[2] > 0)
        return cond

psi_epsilon = psi_epsilon_gen(a=1, name='psi_epsilon',
                                longname='METE individual-energy distribution'
                                )
