import numpy as np
import statistics
import math
'''

    Var from self, numpy, Statistics

        https://www.stechies.com/calculating-variance-standard-deviation-python/

    Ref:
        https://en.wikipedia.org/wiki/Mean_squared_error
        https://stackabuse.com/calculating-variance-and-standard-deviation-in-python/

'''
def variance(val):
    numb = len(val)
    # m will have the mean value
    m = sum(val) / numb
    # Square deviations
    devi = [(x - m) ** 2 for x in val]
    # Variance
    variance = sum(devi) / numb
    return variance

# Finding the variance is essential before calculating the standard deviation
def varinc(val, ddof=0):
    n = len(val)
    m = sum(val) / n
    return sum((x - m) ** 2 for x in val) / (n - ddof)
# finding the standard deviation
def stddev(val):
    vari = varinc(val)
    stdev = math.sqrt(vari)
    return stdev

input = [0.41, 0.25, 0.15, 0.1, 0.06, 0.03]

a = np.array(input)

var = np.var(a)

print('[self]      var =', variance(input))
print('[self]      sd  =', stddev(input))
print('[NUMPY]     var =', var)
print('[NUMPY]     sd  =', np.std(a))
print('[STAT MOD] pvar =', statistics.pvariance(a))
print('[STAT MOD]  var =', statistics.variance(a))
print('[STAT MOD]  psd =', statistics.pstdev(a))
print('[STAT MOD]  sd  =', statistics.stdev(a))