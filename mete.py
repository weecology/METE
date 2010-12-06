"""Module for fitting and testing Harte et al.'s maximum entropy models"""

from __future__ import division
from math import log, exp, isnan
from scipy.optimize import bisect, fsolve
from scipy.stats import logser
from numpy import array, e, empty
import matplotlib.pyplot as plt

# Set the distance from the undefined boundaries of the Lagrangian multipliers
# to set the upper and lower boundaries for the numerical root finders
DIST_FROM_BOUND = 10 ** -15

def get_lambda_sad(S, N, approx='no', version='2009'):
    """Solve for lambda_1 
        
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    approx -- 'no' uses either 2008-eq. B.4 or 2009-eq. 3, which use minimal approximations
              'yes' uses eq. 7b which uses an additional approximation (e^-lambda_1~1-lambda_1)
              the default is 'no'; using the default is strongly recommended
              unless there is a very clear reason to do otherwise.
    version -- '2008': Harte et al. 2008 based on eq.(B.4)
               '2009': Harte et al. 2009 based on eq.(3)
               the default is '2009'; using the default is recommended for
               relatively low values of S
    
    """
    #TO DO: check to see if 'bisect' can be swapped out for 'fsolve'
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    BOUNDS = [0, 1]
    
    # Solve for lambda_sad using the substitution x = e**-lambda_1
    if approx == 'no':    
        if version == '2009':
            m = array(range (1, N+1)) 
            y = lambda x: S / N * sum(x ** m) - sum((x ** m) / m)
            exp_neg_lambda_sad = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 1.005, 
                                        xtol = 1.490116e-08)
        if version == '2008':
            y = lambda x: 1 / log(1 / (1 - x)) * x / (1 - x) - N / S
            exp_neg_lambda_sad = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                                        BOUNDS[1] - DIST_FROM_BOUND)
    else:
        y = lambda x: x * log(1 / x) - S / N
        exp_neg_lambda_sad = fsolve(y, BOUNDS[1] - DIST_FROM_BOUND)
        
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
        exp_neg_lambda_spatialdistrib = bisect(y, BOUNDS[0] + DIST_FROM_BOUND, 
                                               BOUNDS[1] - DIST_FROM_BOUND)
        lambda_spatialdistrib = -1 * log(exp_neg_lambda_spatialdistrib)
    return lambda_spatialdistrib

def get_mete_sad(S, N):
    """Use lambda_1 to generate SAD predicted by the METE
    
    Keyword arguments:
    S -- the number of species
    N -- the total number of individuals
    
    """
    
    assert S > 1, "S must be greater than 1"
    assert N > 0, "N must be greater than 0"
    assert S/N < 1, "N must be greater than S"
    
    lambda_sad = get_lambda_sad(S,N)
    p = e ** -lambda_sad
    abundance  = list(empty([S]))
    rank = range(1, S+1)
    revrank = rank.reverse()
      
    if p >= 1:        
        for i in range(0, S):               
            n = array(range(1, N + 1))
            y = lambda x: (sum(p ** array(range(1, x + 1)) / 
                               array(range(1, x + 1))) / 
                           sum(p ** n / n) - (rank[i]-0.5) / S)
            if y(1) > 0:
                abundance[i] = 1
            else:
                abundance[i] = int(round(bisect(y,1,N)))                
    else:
        for i in range(0, S): 
            y = lambda x: logser.cdf(x,p) / logser.cdf(N,p) - (rank[i]-0.5) / S
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
        n = array(range(1, 2 * N_A + 1)) 
        out.append(x[1] / 2 / N_A * sum(x[0] ** n) - sum(x[0] ** n / n))
        return out
    
    def solve_for_S_2A(S, N):
        x_A = exp(-get_lambda_sad(S, N))
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
    elif math.isnan(S):
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