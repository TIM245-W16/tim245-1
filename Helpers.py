import numpy as np
from math import sqrt, fabs

def rolling_standard_dev(arr, n=5):
    return np.std(rolling_window(arr, n), 1)

#http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#http://bicorner.com/2015/11/16/time-series-analysis-using-ipython/
def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

#http://bicorner.com/2015/11/16/time-series-analysis-using-ipython/
def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

class GarchForecaster:
    def __init__(self, params, horizon=1):
        self.omega, self.alpha, self.beta = params
    
    def forecast(self, sigma_t):

        sigma_tplus1_sqrd = self.omega + \
                            (self.alpha * pow(sigma_t*np.random.normal(),2)) + \
                            (self.beta * pow(sigma_t,2))

        return sqrt(fabs(sigma_tplus1_sqrd))