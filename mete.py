"""Module for fitting and testing Harte et al.'s maximum entropy models

Terminology and notation follows Harte (2011)

"""

from __future__ import division
from math import log, exp, isnan, floor, ceil, factorial
import os.path
import sys

import cPickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, fsolve
from scipy.stats import logser, geom
from numpy.random import random_integers
from random import uniform
from numpy import array, e, empty

from mete_distributions import *



def trunc_logser_pmf(x, p, upper_bound):
    """Probability mass function for the upper truncated log-series

    Parameters
    ----------
    x : array_like
        Values of `x` for which the pmf should be determined
    p : float
        Parameter for the log-series distribution
    upper_bound : float
        Upper bound of the distribution
    
    Returns
    -------
    pmf : array
        Probability mass function for each value of `x`
        
    """
    if p < 1:
        return logser.pmf(x, p) / logser.cdf(upper_bound, p)
    else:
        x = np.array(x)
        ivals = np.arange(1, upper_bound + 1)
        normalization = sum(p ** ivals / ivals)
        pmf = (p ** x / x) / normalization
        return pmf

def trunc_logser_cdf(x_max, p, upper_bound):
    """Cumulative probability function for the upper truncated log-series"""
    if p < 1:
        #If we can just renormalize the untruncated cdf do so for speed
        return logser.cdf(x_max, p) / logser.cdf(upper_bound, p)
    else:
        x_list = range(1, int(x_max) + 1)
        cdf = sum(trunc_logser_pmf(x_list, p, upper_bound))
        return cdf

def trunc_logser_rvs(p, upper_bound, size):
    """Random variates of the upper truncated log-series
    
    Currently this function only supports random variate generation for p < 1.
    This will cover most circumstances, but it is possible to have p >= 1 for
    the truncated version of the distribution.
    
    """
    assert p < 1, 'trunc_logser_rvs currently only supports random number generation for p < 1'
    size = int(size)    
    rvs = logser.rvs(p, size=size)
    for i in range(0, size):
        while(rvs[i] > upper_bound):
            rvs[i] = logser.rvs(p, size=1)
    return rvs

def get_beta(Svals, Nvals, version='precise', beta_dict={}):
    """Solve for Beta, the sum of the two Lagrange multipliers for R(n, epsilon)
        
    Parameters
    ----------
    Svals : int or array_like
        The number of species
    Nvals : int or array_like
        The total number of individuals
    version : {'precise', 'untruncated', 'approx'}, optional
        Determine which solution to use to solve for Beta. The default is
           'precise', which uses minimal approximations.
        'precise' uses minimal approximations and includes upper trunction of
            the distribution at N_0 (eq. 7.27 from Harte et al. 2011)
        'untruncated' uses minimal approximations, but assumes that the
            distribution of n goes to infinity (eq. B.4 from Harte et al. 2008)
        'approx' uses more approximations, but will run substantially faster,
            especially for large N (equation 7.30 from Harte 2011)
    beta_dict : dict, optional
        A dictionary of beta values so that beta can be looked up rather than
        solved numerically. This can substantially speed up execution.
                   
    Both Svals and Nvals can be vectors to allow calculation of multiple values
    of Beta simultaneously. The vectors must be the same length.
    
    Returns
    -------
    betas : list
        beta values for each pair of Svals and Nvals
        
    """
    #Allow both single values and iterables for S and N by converting single values to iterables
    if not hasattr(Svals, '__iter__'):
        Svals = array([Svals])
    else:
        Svals = array(Svals)
    if not hasattr(Nvals, '__iter__'):
        Nvals = array([Nvals])
    else:
        Nvals = array(Nvals)
    
    assert len(Svals) == len(Nvals), "S and N must have the same length"
    assert all(Svals > 1), "S must be greater than 1"
    assert all(Nvals > 0), "N must be greater than 0"
    assert all(Svals/Nvals < 1), "N must be greater than S"
    assert version in ('precise', 'untruncated', 'approx'), "Unknown version provided"
    
    betas = []
    for i, S in enumerate(Svals):
        N = Nvals[i]
        
        # Set the distance from the undefined boundaries of the Lagrangian multipliers
        # to set the upper and lower boundaries for the numerical root finders
        BOUNDS = [0, 1]
        DIST_FROM_BOUND = 10 ** -15
        
        #If not, solve for beta using the substitution x = e**-beta
        if (S, N) in beta_dict:
            betas.append(beta_dict[(S, N)])
        elif version == 'precise':    
            m = array(range(1, int(N)+1)) 
            y = lambda x: sum(x ** m / N * S) - sum((x ** m) / m)
            exp_neg_beta = bisect(y, BOUNDS[0] + DIST_FROM_BOUND,
                                        min((sys.float_info[0] / S) ** (1 / N), 2), xtol = 1.490116e-08)
            betas.append(-1 * log(exp_neg_beta))
        elif version == 'untruncated':
            y = lambda x: 1 / log(1 / (1 - x)) * x / (1 - x) - N / S
            exp_neg_beta = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                                        BOUNDS[1] - DIST_FROM_BOUND)
            betas.append(-1 * log(exp_neg_beta))
        elif version == 'approx':
            y = lambda x: x * log(1 / x) - S / N
            betas.append(fsolve(y, 0.0001))
        
        #Store the value in the dictionary to avoid repeating expensive
        #numerical routines for the same values of S and N. This is
        #particularly important for determining pdfs through mete_distributions.
        beta_dict[(S, N)] = betas[-1]

    #If only a single pair of S and N values was passed, return a float
    if len(betas) == 1:
        betas = betas[0]

    return betas

def get_lambda2(S, N, E):
    """Return lambda_2, the second Lagrangian multiplier for R(n, epsilon) 
    
    lambda_2 is calculated using equation 7.26 from Harte 2011.
    
    """
    return S / (E - N)

def get_lambda1(S, N, E, version='precise', beta_dict={}):
    """Return lambda_1, the first Lagrangian multiplier for R(n, epsilon)
    
    lamba_1 is calculated using equation 7.26 from Harte 2011 and get_beta().
    
    """
    beta = get_beta(S, N, version, beta_dict)
    return beta - get_lambda2(S, N, E)

def get_dict(filename):
    """Check if lookup dictionary for lamba exists. If not, create an empty one.
    Arguments: 
    filename = is the name of the dictionary file to read from, (e.g., 'beta_lookup_table.pck')
    """
    if os.path.exists(filename):
        dict_file = open(filename, 'r')
        dict_in = cPickle.load(dict_file)
        dict_file.close()
        print("Successfully loaded lookup table with %s lines" % len(dict_in))
    else:
        dict_file = open(filename, 'w')
        dict_in = {}
        cPickle.dump(dict_in, dict_file)
        dict_file.close()
        print("No lookup table found. Creating an empty table...")
    return dict_in

def save_dict(dictionary, filename):
    """Save the current beta lookup table to a file
    Arguments:
    dictionary = the dictionary object to output
    filename = the name of the dictionary file to write to (e.g., 'beta_lookup_table.pck')
    """
    dic_output = open(filename, 'w')
    cPickle.dump(dictionary, dic_output)
    dic_output.close()

def build_beta_dict(S_start, S_end, N_max, N_min=1, filename='beta_lookup_table.pck'):
    """Add values to the lookup table for beta
    
    Starting at S_start and finishing at S_end this function will take values
    of N from S + 1 to N_max, determine if a value of beta is already in the 
    lookup table for the current value of values of S and N, and if not then
    calculate the value of beta and add it to the dictionary.
    
    Values are stored for S and N rather than for N/S because the precise form
    of the solution (eq. 7.27 in Harte 2011) depends on N as well as N/S due
    to the upper trunctation of the distribution at N.
    
    """
    beta_dictionary = get_dict(filename)
    for S in range(S_start, S_end + 1):
        N_start = max(S + 1, N_min)
        for N in range(N_start, N_max):
            if (S, N) not in beta_dictionary:
                beta_dictionary[(S, N)] = get_beta(S, N)
    save_dict(beta_dictionary, filename)

def get_mete_pmf(S0, N0, beta = None):
    """Get the truncated log-series PMF predicted by METE"""
    if beta == None:
        beta = get_beta(S0, N0)
    p = exp(-beta)
    truncated_pmf = trunc_logser_pmf(range(1, int(N0) + 1), p, N0)
    return truncated_pmf

def get_mete_sad(S0, N0, beta=None, bin_edges=None):
    """Get the expected number of species with each abundance
    
    If no value is provided for beta it will be solved for using S0 & N0
    If bin_edges is not provided then the values returned are the estimated
        number of species for each integer value from 1 to N0
    If bin_edges is provided it should be an array of bin edges including the
        bottom and top edges. The last value in bin_edge should be > N0
    
    """
    pmf = get_mete_pmf(S0, N0, beta)
    if bin_edges != None:
        N = array(range(1, int(N0) + 1))
        binned_pmf = []
        for edge in range(0, len(bin_edges) - 1):
            bin_probability = sum(pmf[(N >= bin_edges[edge]) &
                                      (N < bin_edges[edge + 1])])
            binned_pmf.append(bin_probability)
        pmf = array(binned_pmf)
    predicted_sad = S0 * pmf
    return predicted_sad
            
def get_lambda_spatialdistrib(A, A0, n0):
    """Solve for lambda_PI from Harte 2011
    
    Keyword arguments:
    A -- the spatial scale of interest
    A0 -- the maximum spatial scale under consideration
    n0 -- the number of individuals of the focal species at scale A0
    
    """
    assert type(n0) is int, "n must be an integer"
    assert A > 0 and A0 > 0, "A and A0 must be greater than 0"
    assert A <= A0, "A must be less than or equal to A0"
    
    y = lambda x: x / (1 - x) - (n0 + 1) * x ** (n0 + 1) / (1 - x ** (n0 + 1)) - n0 * A / A0
    if A < A0 / 2:
        # Set the distance from the undefined boundaries of the Lagrangian multipliers
        # to set the upper and lower boundaries for the numerical root finders
        BOUNDS = [0, 1]
        DIST_FROM_BOUND = 10 ** -15
        exp_neg_lambda = bisect(y, BOUNDS[0] + DIST_FROM_BOUND,
                                    BOUNDS[1] - DIST_FROM_BOUND)
    elif A == A0 / 2:
        #Special case from Harte (2011). See text between Eq. 7.50 and 7.51
        exp_neg_lambda = 1
    else:
        # x can potentially go up to infinity 
        # thus use solution of a logistic equation as the starting point
        exp_neg_lambda = (fsolve(y, - log(A0 / A - 1)))[0] 
    lambda_spatialdistrib = -1 * log(exp_neg_lambda)
    return lambda_spatialdistrib

def get_mete_rad(S, N, beta=None, beta_dict={}):
    """Use beta to generate SAD predicted by the METE
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    beta -- allows input of beta by user if it has already been calculated
    
    """
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    if beta is None:
        beta = get_beta(S, N, beta_dict=beta_dict)
    p = e ** -beta
    abundance  = list(empty([S]))
    rank = range(1, int(S)+1)
    rank.reverse()
      
    if p >= 1:        
        for i in range(0, int(S)):               
            y = lambda x: trunc_logser_cdf(x, p, N) - (rank[i]-0.5) / S
            if y(1) > 0:
                abundance[i] = 1
            else:
                abundance[i] = int(round(bisect(y,1,N)))                
    else:
        for i in range(0, int(S)): 
            y = lambda x: logser.cdf(x,p) / logser.cdf(N,p) - (rank[i]-0.5) / S
            abundance[i] = int(round(bisect(y, 0, N)))
    return (abundance, p)

def get_mete_sad_geom(S, N):
    """METE's predicted RAD when the only constraint is N/S
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    """
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    p = S / N
    abundance  = list(empty([S]))
    rank = range(1, int(S)+1)
    rank.reverse()
                   
    for i in range(0, int(S)): 
        y = lambda x: geom.cdf(x,p) / geom.cdf(N,p) - (rank[i]-0.5) / S
        abundance[i] = int(round(bisect(y, 0, N)))
    return (abundance, p)

def downscale_sar(A, S, N, Amin):
    """Predictions for downscaled SAR using Eq. 7 from Harte et al. 2009"""
    beta = get_beta(S, N)
    x = exp(-beta)
    S = S / x - N * (1 - x) / (x - x ** (N + 1)) * (1 - x ** N / (N + 1))
    A /= 2
    N /= 2
    if A <= Amin:
        return ([A], [S])
    else:
        down_scaled_data = downscale_sar(A, S, N, Amin)
        return (down_scaled_data[0] + [A], down_scaled_data[1] + [S])

def upscale_sar(A, S, N, Amax):
    """Predictions for upscaled SAR using Eqs. 8 and 9 from Harte et al. 2009"""
    def equations_for_S_2A(x, S_A, N_A):
        """Implicit equations for S(2A) given S(A) and N(A)"""
        # TO DO: make this clearer by separating equations and then putting them
        #        in a list for output
        out = [x[1] / x[0] - 2 * N_A *
               (1 - x[0]) / (x[0] - x[0] ** (2 * N_A + 1)) *
               (1 - x[0] ** (2 * N_A) / (2 * N_A + 1)) - S_A]
        n = array(range(1, int(2 * N_A + 1))) 
        out.append(x[1] / 2 / N_A * sum(x[0] ** n) - sum(x[0] ** n / n))
        return out
    
    def solve_for_S_2A(S, N):
        x_A = exp(-get_beta(1.5 * S, 2 * N))
        x0 = fsolve(equations_for_S_2A, [x_A, S], args=(S, N), full_output = 1)
        S_2A, convergence = x0[0][1], x0[2]
        if convergence != 1:
            return float('nan')
        else:
            return S_2A
    
    S = solve_for_S_2A(S, N)
    A *= 2
    N *= 2
    if A >= Amax:
        return ([A], [S])
    elif isnan(S):
        return ([A], S)
    else:
        up_scaled_data = upscale_sar(A, S, N, Amax)
        return ([A] + up_scaled_data[0], [S] + up_scaled_data[1])

def sar(A0, S0, N0, Amin, Amax):
    """Harte et al. 2009 predictions for the species area relationship
    
    Takes a minimum and a maximum area along with the area, richness, and
    abundance at some anchor scale and determines the richness at all bisected
    and/or doubled scales so as to include Amin and Amax.
    
    """
    # This is where we will deal with adding the anchor scale to the results    

def predicted_slope(S, N):
    """Calculates slope of the predicted line for a given S and N
    
    by combining upscaling one level and downscaling one level from 
    the focal scale
    
    """
    ans_lower = downscale_sar(2, S, N, 1)
    if isnan(ans_lower[1][0]) == False:
        S_lower = array(ans_lower[1][0])
        ans_upper = upscale_sar(2, S, N, 4)
        if isnan(ans_upper[1][0]) == True:
            print("Error in upscaling. z cannot be computed.")
            return float('nan')
        else: 
            S_upper = array(ans_upper[1][0])
            return (log(S_upper / S_lower) / 2 / log(2))
    else:
        print("Error in downscaling. Cannot find root.")
        return float('nan')
    
def get_slopes(site_data):
    """get slopes from various scales, output list of area slope and N/S
    
    input data is a list of lists, each list contains [area, mean S, mean N]
    
    """
    # return a list containing 4 values: area, observed slope, predicted slope, and N/S
    # ToDo: figure out why values of S as low as 1 still present problems for 
    #       this approach, this appears to happen when S << N
    data = array(site_data)    
    Zvalues = []
    area = data[:, 0]
    S_values = data[:, 1]
    for a in area:
        if a * 4 <= max(area): #stop at last area
            S_down = float(S_values[area == a])
            S_focal = float(S_values[area == a * 2 ])
            S_up = float(S_values[area == a * 4])
            if S_focal >= 2: #don't calculate if S < 2
                N_focal = float(data[area == a * 2, 2])
                z_pred = predicted_slope(S_focal, N_focal)
                z_emp = (log(S_up) - log(S_down)) / 2 / log(2)
                NS = N_focal / S_focal
                parameters = [a * 2, z_emp, z_pred, NS]
                Zvalues.append(parameters) 
            else:
                continue
        else:
            break
    return Zvalues    
    
def plot_universal_curve(slopes_data):
    """plots ln(N/S) x slope for empirical data and MaxEnt predictions. 
    
    Predictions should look like Harte's universal curve
    input data is a list of lists. Each list contains:
    [area, empirical slope, predicted slope, N/S]
    
    """
    #TO DO: Add argument for axes
    slopes = array(slopes_data)
    NS = slopes[:, 3]
    z_pred = slopes[:, 2]
    z_obs = slopes[:, 1]
    #plot Harte's universal curve from predictions with empirical data to analyze fit
    plt.semilogx(NS, z_pred, 'bo')
    plt.xlabel("ln(N/S)")
    plt.ylabel("Slope")
    plt.hold(True)
    plt.semilogx(NS, z_obs, 'ro')
    plt.show()

def heap_prob(n, A, n0, A0, pdict={}):
    """
    Determines the HEAP probability for n given A, no, and A0
    Uses equation 4.15 in Harte 2011
    Returns the probability that n individuals are observed in a quadrat of area A
    """
    i = int(log(A0 / A,2))
    if (n,n0,i) not in pdict:
        if i == 1:
            pdict[(n,n0,i)] = 1 / (n0 + 1)
        else:
            A = A * 2
            pdict[(n,n0,i)] = sum([heap_prob(q, A, n0, A0, pdict)/ (q + 1) for q in range(n, n0 + 1)])
    return pdict[(n,n0,i)]

def heap_pmf(A, n0, A0):
    """Determines the probability mass function for HEAP
    
    Uses equation 4.15 in Harte 2011
    
    """
    pmf = [heap_prob(n, A, n0, A0) for n in range(0, n0 + 1)]
    return pmf
    
def binomial(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def get_big_binomial(n, k, fdict):
    """returns the natural log of the binomial coefficent n choose k"""
    if n > 0 and k > 0:
        nFact = fdict[n]
        kFact = fdict[k]
        nkFact = fdict[n - k]
        return nFact - kFact - nkFact
    else:
        return 0

def get_heap_dict(n, A, n0, A0, plist=[0, {}]):
    """
    Determines the HEAP probability for n given A, n0, and A0
    Uses equation 4.15 in Harte 2011
    Returns a list with the first element is the probability of n individuals 
    beting observed in a quadrat of area A, and the second element is a 
    dictionary that was built to compute that probability
    """
    i = int(log(A0 / A,2))
    if (n,n0,i) not in plist[1]:
        if i == 1:
            plist[1][(n,n0,i)] = 1 / (n0 + 1)
        else:
            A = A * 2
            plist[1][(n,n0,i)] = sum([get_heap_dict(q, A, n0, A0, plist)[0]/ (q + 1) for q in range(n, n0 + 1)])
    plist[0] = plist[1][(n,n0,i)]
    return plist

def build_heap_dict(n,n0,i, filename='heap_lookup_table.pck'):
    """Add values to the lookup table for heap"""
    heap_dictionary = get_dict(filename)
    if (n,n0,i) not in heap_dictionary:    
        A = 1        
        A0 = 2 ** i
        heap_dictionary.update( get_heap_dict(n,A,n0,A0,[0,heap_dictionary])[1] )
        save_dict(heap_dictionary, filename)
    print("Dictionary building completed")    

def bisect_prob(n, A, n0, A0, psi, pdict={}):
    """Theorem 2.3 in Conlisk et al. (2007)"""
    total = 0
    i = A0 / A 
    a = (1 - psi) / psi  
    if(i == 2):
        pdict[(n, n0, i)] = single_prob(n, A, n0, A0, psi) 
    else:
        A = A * 2
        pdict[(n, n0, i)] = sum([bisect_prob(q, A, n0, A0, psi, pdict) * single_prob(n, A, q, A0, psi) for q in range(n, n0 + 1)])
    return pdict[(n, n0,i )] 


def sim_spatial_one_step(abu_list):
    """Simulates the abundances of species after bisecting one cell. 
    
    Input: species abundances in the original cell. 
    Output: a list with two sublists containing species abundances in the two
    halved cells. 
    Assuming indistinguishable individuals (see Harte et al. 2008). 
    
    """
    abu_half_1 = []
    abu_half_2 = []
    for spp in abu_list:
        if spp == 0:
            abu_half_1.append(0)
            abu_half_2.append(0)
        else:
            abu_1 = random_integers(0, spp)
            abu_half_1.append(abu_1)
            abu_half_2.append(spp - abu_1)
    abu_halves = [abu_half_1, abu_half_2]
    return abu_halves

def sim_spatial_whole(S, N, bisec, transect=False, abu=None, beta=None):
    """Simulates species abundances in all cells given S & N at whole plot
    level and bisection number. 
    
    Keyword arguments:
    S -- the number of species 
    N -- the number of individuals 
    bisec -- the number of bisections to carry out (see Note below) 
    transect -- boolean, if True a 1-dimensional spatial community is
                generated, the default is to generate a spatially 2-dimensional
                community
    abu -- an optional abundance vector that can be supplied for the community
           instead of using log-series random variates
   
    Output: a list of lists, each sublist contains species abundance in one
    cell, and x-y coordinate of the cell.
   
    Note: bisection number 1 corresponds to no bisection (whole plot), and the
    the first actual bisection is along the x-axis
    
    """
    if S == 1:
        abu = [N]
    if abu is None:
        if beta is None:
            p = exp(-get_beta(S, N))
        else:
            p = exp(-beta)
        abu = trunc_logser.rvs(p, N, size=S)
    abu_prev = [[1, 1, array(abu)]]
    bisec_num = 1
    while bisec_num < bisec: 
        abu_new = []
        for cell in abu_prev: 
            x_prev = cell[0]
            y_prev = cell[1]
            abu_new_cell = sim_spatial_one_step(cell[2])
            if(transect):
                cell_new_1 = [x_prev * 2 - 1, y_prev, abu_new_cell[0]]
                cell_new_2 = [x_prev * 2, y_prev, abu_new_cell[1]]
            else:
                if bisec_num % 2 != 0:
                    cell_new_1 = [x_prev * 2 - 1, y_prev, abu_new_cell[0]]
                    cell_new_2 = [x_prev * 2, y_prev, abu_new_cell[1]]
                else:
                    cell_new_1 = [x_prev, y_prev * 2 - 1, abu_new_cell[0]]
                    cell_new_2 = [x_prev, y_prev * 2, abu_new_cell[1]]
            abu_new.append(cell_new_1)
            abu_new.append(cell_new_2)
        abu_prev = abu_new
        bisec_num += 1
    return abu_prev

def sim_spatial_whole_iter(S, N, bisec, coords, n_iter = 10000):
    """Simulates the bisection n_iter times and gets the aggregated species
    richness in plots with given coordinates."""
    max_x = 2 ** ceil((bisec - 1) / 2) 
    max_y = 2 ** floor((bisec - 1) / 2)
    if max(array(coords)[:,0]) > max_x or max(array(coords)[:,1]) > max_y:
        print("Error: Coordinates out of bounds.")
        return float('nan')
    else: 
        i = 1
        S_list = []
        while i <= n_iter:
            abu_list = []
            abu_plot = sim_spatial_whole(S, N, bisec)
            for i_coords in coords:
                for j_cell in abu_plot:
                    if j_cell[0] == i_coords[0] and j_cell[1] == i_coords[1]:
                        abu_list.append(j_cell[2])
                        break
            abu_agg = array(abu_list).sum(axis = 0)
            S_i = sum(abu_agg != 0)
            S_list.append(S_i)
            i += 1
        S_avg = sum(S_list) / len(S_list)
        return S_avg
    
def community_energy_pdf(epsilon, S0, N0, E0):
    lambda1 = get_lambda1()
    lambda2 = get_lambda2()
    gamma = lambda1 + epsilon * lambda2
    exp_neg_gamma = exp(-gamma)
    return S0 / N0 * (exp_neg_gamma / (1 - exp_neg_gamma) ** 2 - 
                      exp_neg_gamma ** N0 / (1 - exp_neg_gamma) *
                      (N0 + exp_neg_gamma / (1 - exp_neg_gamma)))

def which(boolean_list):
    """ Mimics the R function 'which()' and it returns the indics of the
    boolean list that are labeled True """
    return [i for i in range(0, len(boolean_list)) if boolean_list[i]]

def order(num_list):
    """
    This function mimics the R function 'order()' and it carries out a 
    bubble sort on a list and an associated index list.
    The function only returns the index list so that the order of the sorting
    is returned but not the list in sorted order
    Note: [x[i] for i in order(x)] is the same as sorted(x)
    """
    num_list = list(num_list)
    list_length = len(num_list)
    index_list = range(0, list_length)
    swapped = True 
    while swapped:
        swapped = False
        for i in range(1, list_length):
            if num_list[i-1] > num_list[i]:
                temp1 = num_list[i-1]
                temp2 = num_list[i]
                num_list[i-1] = temp2
                num_list[i] = temp1 
                temp1 = index_list[i-1]
                temp2 = index_list[i]
                index_list[i-1] = temp2
                index_list[i] = temp1 
                swapped = True
    return index_list


def single_rvs(n0, psi, size=1):
    """Generate random deviates from the single division model, still 
    is not working properly possibily needs to be checked"""
    cdf = single_cdf(1,n0,2,psi)
    xvals = [0] * size
    for i in range(size):
        rand_float = uniform(0,1)
        temp_cdf = list(cdf + [rand_float])
        ordered_values = order(temp_cdf)
        xvals[i] = [j for j in range(0,len(ordered_values)) if ordered_values[j] == (n0 + 1)][0]
    return xvals 

def getF(a, n):
    """ Eq. 7 in Conlisk et al. (2007) """
    out = 1  
    if n != 0:
        for i in range(1, n + 1):
            out *= (a + i - 1) / i  
    return out 

def single_prob(n, A, n0, A0, psi):
    """
    Eq. 1.3 in Conlisk et al. (2007), note that this implmentation is
    only correct when the variable c = 2 
    """
    a = (1 - psi) / psi 
    return (getF(a, n) * getF(a, n0 - n)) / getF(2 * a, n0) 


def single_cdf(A, n0, A0, psi):
    cdf = [0.0] * (n0 + 1) 
    for n in range(0, n0 + 1):
        if n == 0:
            cdf[n] = single_prob(n, A, n0, A0, psi) 
        else:
            cdf[n] = cdf[n - 1] + single_prob(n, A, n0, A0, psi) 
    return cdf
