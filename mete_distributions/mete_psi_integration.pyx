from __future__ import division
import mpmath
import math

def get_integral_for_psi_cdf(double x, double beta, double lambda2, int N0):
    cdef double int_start = math.exp(-beta)
    cdef double int_end = math.exp(-(beta + (x - 1) * lambda2))
    def partial_integral(double x):
        return (-N0 * x ** (N0 + 1) + (N0 + 1) * x ** N0) / (1 - x) ** 2
    return float(mpmath.quad(partial_integral, [int_start, int_end]))