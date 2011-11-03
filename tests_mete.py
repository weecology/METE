"""Tests for the METE (Maximum Entropy Theory of Ecology) Module"""

from mete import *
import nose
from nose.tools import assert_almost_equals, assert_equals
from decimal import Decimal

def test_get_lambda_sad():
    """Tests SAD lambda estimates against values from Table 7.2 of Harte 2011
    
    The table of test values is structured as S0, N0, Beta
    
    """
    data = [[4, 16, '0.0459'],
            [4, 64, '-0.00884'],
            [4, 1024, '-0.00161'],
            [4, 16384, '-0.000135'],
            [16, 64, '0.101'],
            [16, 256, '0.0142'],
            [16, 4096, '0.000413'],
            [16, 65536, '0.0000122'],
            [64, 256, '0.102'],
            [64, 1024, '0.0147'],
            [64, 16384, '0.000516'],
            [64, 262144, '0.0000228']]
    for line in data:
        yield check_get_lambda_sad, line[0], line[1], line[2]
        
def check_get_lambda_sad(S0, N0, beta_known):
    beta_code = get_lambda_sad(S0, N0)
    
    #Determine number of decimal places in known value and round code value equilalently
    decimal_places_in_beta_known = abs(Decimal(beta_known).as_tuple().exponent)
    beta_code_rounded = round(beta_code, decimal_places_in_beta_known)
    
    assert_almost_equals(beta_code_rounded, float(beta_known), places=6)
    
if __name__ == "__main__":
    nose.run()