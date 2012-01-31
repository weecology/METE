"""Tests for the METE (Maximum Entropy Theory of Ecology) Module"""

from __future__ import division
from mete import *
import nose
from nose.tools import assert_almost_equals, assert_equals
from decimal import Decimal
from math import exp

#Fitted values of parameters from Harte 2011. The fits were done using different
#software and therefore represent reasonable tests of the implemented algorithms
#Numbers with decimal places are entered as strings to allow Decimal to properly handle them
table7pt2 = [[4, 16, 64, '0.0459', '0.116', '5.4', '-0.037', '0.083', '0.74'],
             [4, 64, 256, '-0.00884', '0.0148', '5.3', '-0.030', '0.021', '-0.56'],
             [4, 1024, 4096, '-0.00161', '0.000516', '5.3', '-0.0029', '0.0013', '-1.6'],
             [4, 16384, 65536, '-0.000135', '0.0000229', '5.3', '-0.00022', '0.000081', '-2.3'],
             [4, 16, 16384, '0.0459', '0.116', '4.0', '0.046', '0.00024', '0.74'],
             [4, 64, 65536, '-0.00884', '0.0148', '4.0', '-0.0089', '0.000061', '-0.6'],
             [4, 1024, 1048576, '-0.00161', '0.000516', '4.0', '-0.0016', '0.00000382', '-1.6'],
             [4, 16384, 16777216, '-0.000135', '0.0000229', '4.0', '-0.00014', '0.000000239', '-2.3'],
             [16, 64, 256, '0.101', '0.116', '21.4', '0.018', '0.083', '6.4'],
             [16, 256, 1024, '0.0142', '0.0148', '21.3', '-0.0066', '0.021', '3.6'],
             [16, 4096, 16384, '0.000413', '0.000516', '21.3', '-0.00089', '0.0013', '1.7'],
             [16, 65536, 262144, '0.0000122', '0.0000229', '21.3', '-0.000069', '0.000081', '0.79'],
             [16, 64, 65536, '0.101', '0.116', '16.1', '0.10', '0.00024', '6.4'],
             [16, 256, 262144, '0.0142', '0.0148', '16.0', '0.014', '0.000061', '3.6'],
             [16, 4096, 4194304, '0.000413', '0.000516', '16.0', '0.00041', '0.00000382', '1.7'],
             [16, 65536, 67108864, '0.0000122', '0.0000229', '16.0', '0.000012', '0.000000239', '0.79'],
             [64, 256, 1024, '0.102', '0.116', '85.4', '0.018', '0.083', '26'],
             [64, 1024, 4096, '0.0147', '0.0148', '85.3', '-0.0061', '0.021', '15'],
             [64, 16384, 65536, '0.000516', '0.000516', '85.3', '-0.00079', '0.0013', '8.5'],
             [64, 262144, 1048576, '0.0000228', '0.0000229', '85.3', '-0.000059', '0.000081', '6.0'],
             [64, 256, 262144, '0.102', '0.116', '64.2', '0.10', '0.00024', '26'],
             [64, 1024, 1048576, '0.0147', '0.0148', '64.1', '0.015', '0.000062', '15'],
             [64, 16384, 16777216, '0.000516', '0.000516', '64.1', '0.00051', '0.00000382', '8.5'],
             [64, 262144, 268435456, '0.0000228', '0.0000229', '64.1', '0.000023', '0.000000239', '6.0'],
             [256, 1024, 4096, '0.102', '0.116', '341.4', '0.018', '0.083', '102'],
             [256, 4096, 16384, '0.0147', '0.0148', '341.3', '-0.0061', '0.021', '61'],
             [256, 65536, 262144, '0.000516', '0.000516', '341.3', '-0.00079', '0.0013', '34'],
             [256, 1048576, 4194304, '0.0000228', '0.0000229', '341.3', '-0.000059', '0.000081', '24'],
             [256, 1024, 1048576, '0.102', '0.116', '256.4', '0.10', '0.00024', '102'],
             [256, 4096, 4194304, '0.0147', '0.0148', '256.3', '0.015', '0.000062', '61'],
             [256, 65536, 67108864, '0.000516', '0.000516', '256.3', '0.00051', '0.00000382', '34'],
             [256, 1048576, 1073741824, '0.0000228', '0.0000229', '256.3', '0.000023', '0.000000239', '24']]

table7pt4 = [[1, 1/4,  '0.333'],
             [1, 1/8,  '0.125'],
             [1, 1/16, '0.067'],
             [2, 1/4,  '0.434'],
             [2, 1/8,  '0.220'],
             [2, 1/16, '0.115'],
             [4, 1/4,  '0.568'],
             [4, 1/8,  '0.344'],
             [4, 1/16, '0.201'],
             [32, 1/4, '0.901'],
             [32, 1/8, '0.801'],
             [32, 1/16, '0.667']]

def test_get_lambda_sad_precise():
    """Tests SAD lambda estimates using the 'precise' method against values
    from Table 7.2 of Harte 2011
    
    The table of test values is structured as S0, N0, Beta
    
    """
    data = set([(line[0], line[1], line[3]) for line in table7pt2])
    for line in data:
        yield check_get_lambda_sad, line[0], line[1], 'precise', line[2]
        
def test_get_lambda_sad_multiplevalues():
    """Check that multiple values are handled properly"""
    Svals = [16, 256]
    Nvals = [64, 4096]
    betavals = get_lambda_sad(Svals, Nvals)
    assert_almost_equals(betavals[0], 0.101, places=3)
    assert_almost_equals(betavals[1], 0.0147, places=4)
    
def test_get_lambda_sad_approx():
    """Tests SAD lambda estimates using the 'approx' method against values
    from Table 7.2 of Harte 2011
    
    The table of test values is structured as S0, N0, Beta
    
    """
    data = set([(line[0], line[1], line[4]) for line in table7pt2])
    for line in data:
        yield check_get_lambda_sad, line[0], line[1], 'approx', line[2]
        
def check_get_lambda_sad(S0, N0, version, beta_known):
    beta_code = get_lambda_sad(S0, N0, version=version)
    
    #Determine number of decimal places in known value and round code value equilalently
    decimal_places_in_beta_known = abs(Decimal(beta_known).as_tuple().exponent)
    beta_code_rounded = round(beta_code, decimal_places_in_beta_known)
    assert_almost_equals(beta_code_rounded, float(beta_known), places=6)
    
def test_get_lambda1():
    """Tests SAD lambda1 estimates against values from Table 7.2 of Harte 2011
    
    The table of test values is structured as S0, N0, E0, lambda1
    
    """
    data = set([(line[0], line[1], line[2], line[6]) for line in table7pt2])
    for line in data:
        yield check_get_lambda1, line[0], line[1], line[2], line[3]

def check_get_lambda1(S0, N0, E0, lambda1_known):
    lambda1_code = get_lambda1(S0, N0, E0)
    decimal_places_in_lambda1_known = abs(Decimal(lambda1_known).as_tuple().exponent)
    lambda1_code_rounded = round(lambda1_code, decimal_places_in_lambda1_known)
    assert_almost_equals(lambda1_code_rounded, float(lambda1_known), places=6)

def test_get_lambda2():
    """Tests SAD lambda1 estimates against values from Table 7.2 of Harte 2011
    
    The table of test values is structured as S0, N0, E0, lambda2
    
    """
    data = set([(line[0], line[1], line[2], line[7]) for line in table7pt2])
    for line in data:
        yield check_get_lambda2, line[0], line[1], line[2], line[3]

def check_get_lambda2(S0, N0, E0, lambda2_known):
    lambda2_code = get_lambda2(S0, N0, E0)
    decimal_places_in_lambda2_known = abs(Decimal(lambda2_known).as_tuple().exponent)
    lambda2_code_rounded = round(lambda2_code, decimal_places_in_lambda2_known)
    assert_almost_equals(lambda2_code_rounded, float(lambda2_known), places=6)

def test_get_lambda_spatialdistrib_precise():
    """Tests Spatial Abundance Distribution lambda estimates
    
    Test against values from Table 7.4 of Harte 2011 using eq. 7.50
    
    The table of test values is structured as n0, A/A0
    
    """
    data = table7pt4
    for line in data:
        yield check_get_lambda_spatialdistrib, line[1], 1, line[0], line[2]
        
def check_get_lambda_spatialdistrib(A, A0, n0, x_known):
    lambda_code = get_lambda_spatialdistrib(A, A0, n0)
    x_code = exp(-lambda_code)
    
    #Determine number of decimal places in known value and round code value equilalently
    decimal_places_in_x_known = abs(Decimal(x_known).as_tuple().exponent)
    x_code_rounded = round(x_code, decimal_places_in_x_known)
    assert_almost_equals(x_code_rounded, float(x_known), places=3)
    
if __name__ == "__main__":
    nose.run()