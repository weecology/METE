"""Module for fitting and testing Harte et al.'s maximum entropy models"""

#TODO: 1.modifying naming conventions to match Harte 2011
#      2.transition to 'import numpy as np'


from __future__ import division
from math import log, exp, isnan, floor, ceil
from scipy.optimize import bisect, fsolve
from scipy.stats import logser, geom
from numpy.random import random_integers
from numpy import array, e, empty
import matplotlib.pyplot as plt
import os.path
import cPickle
import sys
import numpy as np

def trunc_logser_pmf(x, p, upper_bound):
    """Probability mass function for the upper truncated log-series"""
    x = np.array(x)
    ivals = np.arange(1, upper_bound + 1)
    normalization = sum(p ** ivals / ivals)
    pmf = (p ** x / x) / normalization
    return pmf

def trunc_logser_cdf(x_max, p, upper_bound):
    """Cumulative probability function for the upper truncated log-series"""
    x_list = range(1, floor(x_max) + 1)
    cdf = sum(trunc_logser_pmf(x_list, p, upper_bound))
    return cdf

def get_lambda_sad(S, N, version='precise', lambda_dict={}):
    """Solve for Beta, the sum of the two Lagrange multipliers for R(n, epsilon)
        
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    version -- 'precise':     uses equation 7.27 from Harte 2011, which uses minimal
                              approximations and includes upper trunction of the
                              distribution at N_0
               'untruncated': uses equation B.4 from Harte et al. 2008, which
                              uses minimal approximations, but assumes that the
                              distribution of n goes to infinity
               'approx':      uses equation 7.30 from Harte 2011, which makes more
                              approximations, but will run substantially faster,
                              especially for large N
               the default is 'precise'; using the default is recommended unless
               there is a good reason to do otherwise.
    lambda_dict -- optionally pass in a dictionary of lambda values so that
                   lambda can be looked up rather than solved numerically. This
                   can substantially speed up execution and is highly
                   recommended if large numbers of calculations are being
                   conducted.
    """
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    assert version in ('precise', 'untruncated', 'approx'), "Unknown version provided"
    
    # Set the distance from the undefined boundaries of the Lagrangian multipliers
    # to set the upper and lower boundaries for the numerical root finders
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15
    
    # Solve for lambda_sad using the substitution x = e**-lambda_1
    if version == 'precise':    
        if N / S in lambda_dict:
            return lambda_dict[N / S]
        else:
            m = array(range(1, int(N)+1)) 
            y = lambda x: sum(x ** m / N * S) - sum((x ** m) / m)
            exp_neg_lambda_sad = bisect(y, BOUNDS[0] + DIST_FROM_BOUND,
                                    min((sys.float_info[0] / S) ** (1 / N), 2), xtol = 1.490116e-08)
    elif version == 'untruncated':
        y = lambda x: 1 / log(1 / (1 - x)) * x / (1 - x) - N / S
        exp_neg_lambda_sad = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                                    BOUNDS[1] - DIST_FROM_BOUND)
    elif version == 'approx':
        y = lambda x: x * log(1 / x) - S / N
        exp_neg_lambda_sad = fsolve(y, BOUNDS[1] - DIST_FROM_BOUND)            
            
    lambda_sad = -1 * log(exp_neg_lambda_sad)
    return lambda_sad

def get_lambda_dict(filename='lambda_library.pck'):
    """Check if lookup dictionary for lamba exists. If not, create an empty one."""
    if os.path.exists(filename):
        dict_file = open(filename, 'r')
        dict_lambda = cPickle.load(dict_file)
        dict_file.close()
    else:
        dict_file = open(filename, 'w')
        dict_lambda = {}
        cPickle.dump(dict_lambda, dict_file)
        dict_file.close()
    return dict_lambda

def save_lambda_dict(lambda_dictionary, filename='lambda_library.pck'):
    """Save the current lambda lookup table to a file"""
    dic_output = open(filename, 'w')
    cPickle.dump(lambda_dictionary, dic_output)
    dic_output.close()

def build_lambda_dict(S_start, S_end, N_max, N_min=1, filename='lambda_library.pck'):
    """Add values to the lookup table for beta
    
    Starting at S_start and finishing at S_end this function will take values
    of N from S + 1 to N_max, determine if a value of beta is already in the 
    lookup table for the current value of N/S, and if not then calculate the
    value of beta and add it to the dictionary
    
    """
    lambda_dictionary = get_lambda_dict(filename)
    for S in range(S_start, S_end + 1):
        N_start = max(S + 1, N_min)
        for N in range(N_start, N_max):
            N_over_S = N / S
            if N_over_S not in lambda_dictionary:
                lambda_dictionary[N_over_S] = get_lambda_sad(S, N)
    save_lambda_dict(lambda_dictionary, filename)

def get_mete_pmf(S_0, N_0, lambda_sad = None):
    """Get the truncated log-series PMF predicted by METE"""
    if lambda_sad == None:
        lambda_sad = get_lambda_sad(S_0, N_0)
    p = exp(-lambda_sad)
    truncated_pmf = trunc_logser_pmf(range(1, int(N_0) + 1), p, N_0)
    return truncated_pmf

def get_mete_sad(S_0, N_0, lambda_sad=None, bin_edges=None):
    """Get the expected number of species with each abundance
    
    If no value is provided for lambda_sad it will be solved for using S_0 & N_0
    If bin_edges is not provided then the values returned are the estimated
        number of species for each integer value from 1 to N_0
    If bin_edges is provided it should be an array of bin edges including the
        bottom and top edges. The last value in bin_edge should be > N_0
    
    """
    pmf = get_mete_pmf(S_0, N_0, lambda_sad)
    if bin_edges != None:
        N = array(range(1, int(N_0) + 1))
        binned_pmf = []
        for edge in range(0, len(bin_edges) - 1):
            bin_probability = sum(pmf[(N >= bin_edges[edge]) &
                                      (N < bin_edges[edge + 1])])
            binned_pmf.append(bin_probability)
        pmf = array(binned_pmf)
    predicted_sad = S_0 * pmf
    return predicted_sad
            
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
    
    # Set the distance from the undefined boundaries of the Lagrangian multipliers
    # to set the upper and lower boundaries for the numerical root finders
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15
    
    if A == A_0 / 2:
        # Special case where A = A_0/2 from Harte et al. 2009 after eq. 6
        lambda_spatialdistrib = 0
    else:
        # Solve for lambda_P using the substitution x = e**-lambda_P
        y = lambda x: 1 / (1 - x ** (n_0 + 1)) * (x / (1 - x) - x ** (n_0 +1) *
                                                  (n_0 + 1 / (1 - x))) - (n_0 * A /
                                                                          A_0)
        exp_neg_lambda_spatialdistrib = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                                               BOUNDS[1] - DIST_FROM_BOUND)
        lambda_spatialdistrib = -1 * log(exp_neg_lambda_spatialdistrib)
    return lambda_spatialdistrib

def get_mete_rad(S, N, lambda_sad=None, lambda_dict={}):
    """Use lambda_1 to generate SAD predicted by the METE
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    lambda_sad -- allows input of lambda by user if it has already been calculated
    
    """
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    if lambda_sad is None:
        lambda_sad = get_lambda_sad(S, N, lambda_dict=lambda_dict)
    p = e ** -lambda_sad
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
        x_A = exp(-get_lambda_sad(1.5 * S, 2 * N))
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

def sar(A_0, S_0, N_0, Amin, Amax):
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
            print "Error in upscaling. z cannot be computed."
            return float('nan')
        else: 
            S_upper = array(ans_upper[1][0])
            return (log(S_upper / S_lower) / 2 / log(2))
    else:
        print "Error in downscaling. Cannot find root."
        return float('nan')
    
def get_slopes(site_data):
    """get slopes from various scales, output list of area slope and N/S
    
    input data is a list of lists, each list contains [area, mean S, mean N]
    
    """
    # return a list containing 4 values: area, observed slope, predicted slope, and N/S
    data = array(site_data)    
    Zvalues = []
    area = data[:, 0]
    S_values = data[:, 1]
    for a in area:
        if a * 4 <= max(area): #stop at last area
            S_down = float(S_values[area == a])
            S_focal = float(S_values[area == a * 2 ])
            S_up = float(S_values[area == a * 4])
            if S_focal >= 5: #don't calculate if S < 5
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
    
def sim_spatial_one_step(abd_list):
    """Simulates the abundances of species after bisecting one cell. 
    
    Input: species abundances in the original cell. 
    Output: a list with two sublists containing species abundances in the two halved cells. 
    Assuming indistinguishable individuals (see Harte et al. 2008). 
    
    """
    abd_half_1 = []
    abd_half_2 = []
    for spp in abd_list:
        if spp == 0:
            abd_half_1.append(0)
            abd_half_2.append(0)
        else:
            abd_1 = random_integers(0, spp)
            abd_half_1.append(abd_1)
            abd_half_2.append(spp - abd_1)
    abd_halves = [abd_half_1, abd_half_2]
    return abd_halves

def sim_spatial_whole(S, N, bisec):
    """Simulates species abundances in all cells given S & N at whole plot level and bisection number. 
    
    Output: a list of lists, each sublist contains species abundance in one cell, and x-y coordinate of the cell.
   
    Note: bisection number 1 corresponds to no bisection (whole plot). 
    
    """
    abd_pred = get_mete_rad(S, N, )[0]
    abd_prev = [[1, 1, array(abd_pred)]]
    bisec_num = 1
    while bisec_num < bisec: 
        abd_new = []
        for cell in abd_prev: 
            x_prev = cell[0]
            y_prev = cell[1]
            abd_new_cell = sim_spatial_one_step(cell[2])
            if bisec_num % 2 != 0: 
                cell_new_1 = [x_prev * 2 - 1, y_prev, abd_new_cell[0]]
                cell_new_2 = [x_prev * 2, y_prev, abd_new_cell[1]]
            else:
                cell_new_1 = [x_prev, y_prev * 2 - 1, abd_new_cell[0]]
                cell_new_2 = [x_prev, y_prev * 2, abd_new_cell[1]]
            abd_new.append(cell_new_1)
            abd_new.append(cell_new_2)
        abd_prev = abd_new
        bisec_num += 1
    return abd_prev

def sim_spatial_whole_iter(S, N, bisec, coords, n_iter = 10000):
    """Simulates the bisection n_iter times and gets the aggregated species richness in plots with given coordinates."""
    max_x = 2 ** ceil((bisec - 1) / 2) 
    max_y = 2 ** floor((bisec - 1) / 2)
    if max(array(coords)[:,0]) > max_x or max(array(coords)[:,1]) > max_y:
        print "Error: Coordinates out of bounds."
        return float('nan')
    else: 
        i = 1
        S_list = []
        while i <= n_iter:
            abd_list = []
            abd_plot = sim_spatial_whole(S, N, bisec)
            for i_coords in coords:
                for j_cell in abd_plot:
                    if j_cell[0] == i_coords[0] and j_cell[1] == i_coords[1]:
                        abd_list.append(j_cell[2])
                        break
            abd_agg = array(abd_list).sum(axis = 0)
            S_i = sum(abd_agg != 0)
            S_list.append(S_i)
            i += 1
        S_avg = sum(S_list) / len(S_list)
        return S_avg