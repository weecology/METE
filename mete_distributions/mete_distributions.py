"""Distributions for use with METE"""

from __future__ import division
from macroeco_distributions import *
from mete import *
import numpy as np
import mpmath
import scipy
from scipy.stats import logser, geom, rv_discrete, rv_continuous
from scipy.optimize import bisect
from math import exp
from mete_psi_integration import get_integral_for_psi_cdf
                                      
class trunc_logser_gen(rv_discrete):
    """Upper truncated logseries distribution
    
    Scipy based distribution class for the truncated logseries pmf, cdf and rvs
    
    Usage:
    PMF: trunc_logser.pmf(list_of_xvals, p, upper_bound)
    CDF: trunc_logser.cdf(list_of_xvals, p, upper_bound)
    Random Numbers: trunc_logser.rvs(p, upper_bound, size=1)
    
    """
    
    def _pmf(self, x, p, upper_bound):
        x = np.array(x)
        ivals = np.arange(1, upper_bound + 1)
        normalization = sum(p ** ivals / ivals)
        pmf = (p ** x / x) / normalization
        return pmf

trunc_logser = trunc_logser_gen(a=1, name='trunc_logser',
                                longname='Upper truncated logseries',
                                shapes="upper_bound",
                                extradoc="""Truncated logseries
                                
                                Upper truncated logseries distribution
                                """
                                )

class psi_epsilon:
    """Inidividual-energy distribution predicted by METE (modified from equation 7.24)
    
    lower truncated at 1 and upper truncated at E0.
    
    Methods:
    pdf - probability density function
    cdf - cumulative density function
    ppf - inverse cdf
    rvs - random number generator
    E - first moment (mean)
    
    """
    def __init__(self, S0, N0, E0):
        self.a, self.b = 1, E0
        self.N0 = N0
        self.beta = get_beta(S0, N0)
        self.lambda2 = get_lambda2(S0, N0, E0)
        self.sigma = self.beta + (E0 - 1) * self.lambda2
        self.norm_factor = self.lambda2 / ((exp(-self.beta) - exp(-self.beta * (N0 + 1))) / (1 - exp(-self.beta)) - 
                                (exp(-self.sigma) - exp(-self.sigma * (N0 + 1))) / (1 - exp(-self.sigma)))

    def pdf(self, x):
        exp_neg_gamma = exp(-(self.beta + (x - 1) * self.lambda2))
        if self.a <= x <= self.b:
            return self.norm_factor * exp_neg_gamma * (1 - (self.N0 + 1) * exp_neg_gamma ** self.N0 +
                                                  self.N0 * exp_neg_gamma ** (self.N0 + 1)) / (1 - exp_neg_gamma) ** 2
        else: return 0
        #Below is the exact form of equation 7.24, which seems to contain an error: 
        #return norm_factor * (exp_neg_gamma / (1 - exp_neg_gamma) ** 2 - 
                              #exp_neg_gamma ** N0 / (1 - exp_neg_gamma) *
                              #(N0 + exp_neg_gamma / (1 - exp_neg_gamma)))

    def cdf(self, x):
        int_start = exp(-self.beta)
        int_end = exp(-(self.beta + (x - 1) * self.lambda2))
        return self.norm_factor / self.lambda2 * (1 / (1 - int_start) - 1 / (1 - int_end) + \
                                                  float(get_integral_for_psi_cdf(x, self.beta, self.lambda2, self.N0)))
    def ppf(self, q):
        y = lambda t: self.cdf(t) - q
        x = bisect(y, self.a, self.b, xtol = 1.490116e-08)
        return x
        
    def rvs(self, size):
        out = []
        rand_list = scipy.stats.uniform.rvs(size = size)
        for rand_num in rand_list:
            out.append(self.ppf(rand_num))
        return out
    
    def E(self):
        def mom_1(x):
            return x * self.pdf(x)
        return float(mpmath.quad(mom_1, [self.a, self.b]))

class theta_epsilon:
    """Intraspecific energy/mass distribution predicted by METE (Eqn 7.25)
    
    lower truncated at 1 and upper truncated at E0.
    
    Methods:
    pdf - probability density function
    cdf - cumultaive density function
    ppf - inverse cdf
    rvs - random number generator
    E - first moment (mean)
    
    """
    def __init__(self, S0, N0, E0):
        self.a, self.b = 1, E0
        self.beta = get_beta(S0, N0)
        self.lambda2 = get_lambda2(S0, N0, E0)
        self.lambda1 = self.beta - self.lambda2
        self.sigma = self.beta + (E0 - 1) * self.lambda2
 
    def pdf(self, x, n):
        pdf = self.lambda2 * n * exp(-(self.lambda1 + 
                                       self.lambda2 * x) * n) / (exp(-self.beta * n) - 
                                                                 exp(-self.sigma * n))
        return pdf
    
    def logpdf(self, x, n):
        logpdf = np.log(self.lambda2) + np.log(n) - (self.lambda1 + self.lambda2 * x) * n - \
            np.log(exp(-self.beta * n) - exp(-self.sigma * n))
        return logpdf
    
    def cdf(self, x, n):
        def pdf_n(x):
            return self.pdf(x, n)
        cdf = mpmath.quad(pdf_n, [1, x])
        return float(cdf) 
    
    def ppf(self, n, q):
        y = lambda t: self.cdf(t, n) - q
        x = bisect(y, self.a, self.b, xtol = 1.490116e-08)
        return x
    
    def rvs(self, n, size):
        out = []
        for i in range(size):
            rand_exp = trunc_expon.rvs(self.lambda2 * n, 1)
            while rand_exp > self.b:
                rand_exp = trunc_expon.rvs(self.lambda2 * n, 1)
            out.append(rand_exp)
        return out
        
    def E(self, n):
        def mom_1(x):
            return x * self.pdf(x, n)
        return float(mpmath.quad(mom_1, [self.a, self.b]))

class sad_agsne_gen(rv_discrete):
    """SAD predicted by AGSNE. Right now only takes single input values but not a list. 
    
    Usage:
    pmf: sad_agsne.pmf(x, lambda1, beta, upper_bound)
    cdf: sad_agsne.cdf(x, lambda1, beta, upper_bound)
    
    """
    
    def _pmf(self, x, lambda1, beta, upper_bound):
        ivals = np.arange(1, upper_bound + 1)
        normalization = sum(np.exp(-lambda1 - beta * ivals) / ivals / (1 - np.exp(-lambda1 - beta * ivals)))
        pmf = np.exp(-lambda1 - beta * x) / x / (1 - np.exp(-lambda1 - beta * x)) / normalization
        return pmf
    
    def _cdf(self, x, lambda1, beta, upper_bound):
        cdf = sum([self.pmf(t, lambda1, beta, upper_bound) for t in range(1, x + 1)])
        return cdf
    
    def _argcheck(self, *args):
        self.a = 1
        self.b = args[2]
        cond = (args[0] > 0) & (args[1] > 0) & (args[2] > 1)
        return cond        

sad_agsne = sad_agsne_gen(name='sad_agsne', longname='SAD of AGSNE', 
                          shapes="lambda1, beta, upper_bound")

class psi_agsne:
    """ISD predicted by AGSNE, following S-40 in Harte et al. 2015, lower truncated at 1 and upper truncated at E0.
    
    Methods:
    pdf - probability density function
    cdf - cumulative density function
    ppf - inverse cdf
    rvs - random number generator
    
    Usage:
    psi = psi_agsne([G, S, N, E], [lambda1, beta, lambda3, z])
    psi.pdf(x)
    psi.cdf(x)
    psi.ppf(q) (0 <= q <= 1)
    psi.rvs(size)
    """
    def __init__(self, statvar, pars):
        self.G, self.S, self.N, self.E = statvar
        self.a, self.b = 1, self.E
        self.lambda1, self.beta, self.lambda3, self.z = pars
        self.lambda2 = self.beta - self.lambda3
        self.exp_neg_lambda1 = exp(-self.lambda1)
        self.norm = 1 /  self.lambda3 * sum([exp(-self.lambda1 * x) * (1 / (1 - exp(-self.beta * x)) - 1 / (1 - exp(-(self.lambda2 + self.E * self.lambda3) * x))) for x in range(1, self.S + 1)])
  
    def pdf(self, x):
        exp_neg_gamma = exp(-(self.lambda2 + x * self.lambda3))
        if self.a <= x <= self.b:
            sum_list = [m * (exp_neg_gamma *self.exp_neg_lambda1) ** m / (1 - exp_neg_gamma ** m) ** 2 for m in range(1, S + 1)]
            return  sum(sum_list) / self.norm
        else: return 0

    def cdf(self, x):
        unscaled_cdf = 1 /  self.lambda3 * \
            sum([exp(-self.lambda1 * m) * (1 / (1 - exp(-self.beta * m)) - 1 / (1 - exp(-(self.lambda2 + x * self.lambda3) * m))) for m in range(1, self.S + 1)])
        if self.a <= x <= self.b:
            return unscaled_cdf / self.norm
        elif x < self.a: return 0
        else: return 1
        
    def ppf(self, q):
        y = lambda t: self.cdf(t) - q
        x = bisect(y, self.a, self.b, xtol = 1.490116e-08) 
        return x
    
    def rvs(self, size):
        out = []
        rand_list = scipy.stats.uniform.rvs(size = size)
        for rand_num in rand_list:
            out.append(self.ppf(rand_num))
        return np.array(out)
