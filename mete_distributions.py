"""Distributions for use with METE"""

import numpy as np
from scipy.stats import logser, geom, rv_discrete

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