# This is some beginning is buidling my own models/simulations.
# It wasn't required to do the project, but it may be worked on so it can be included.
import numpy as np

class AR:
    def __init__(self, phi, constant=0.0):
        self.phi_length = len(phi)
        self.phi = phi
        self.constant = constant

    def generate_periods(self,periods):
        """
        AR(periods): y_t = constant + sum_from_i_to_p(phi_i * y_{t-i}) + white_noise
        """        
        series = np.zeros((periods,),dtype=float)
        white_noise = np.random.normal(size=periods)
        for i in range(periods):
            if(i <= self.phi_length):
                series[i,] = np.vdot(self.phi[:i],series[:i]) + self.constant + white_noise[i,]            
            else:
                series[i,] = np.vdot(self.phi,series[(i-self.phi_length):i]) + self.constant + white_noise[i,]

        return series

class MA:
    def __init__(self, theta, constant = 0.0):
        self.theta_length = len(theta)
        self.theta = theta
        self.constant = constant
    
    def generate_periods(self,periods):
        """
        MA(periods): y_t = constant + white_noise_t + sum_from_i_to_p(theta_i * white_noise_{t-i})
        """
        series = np.zeros((periods,),dtype=float)
        white_noise = np.random.normal(size=periods)
        for i in range(periods):
            if(i < self.theta_length):
                series[i,] = np.vdot(self.theta[:i],white_noise[:i]) + white_noise[i,] + self.constant
            else:
                series[i,] = np.vdot(self.theta,series[(i-self.theta_length):i]) + white_noise[i,] + self.constant
        
        return series

class ARMA:
    """
        ARMA(phi,theta) = AR(phi) + MA(theta) + constant + white_noise
    """
    def __init__(self, phi, theta, constant = 0.0):
        self.phi = phi
        self.theta = theta
        self.constant = constant

    def generate_periods(self, periods):
        # the constant should only be added in once so only pass it to AR
        ar = AR(self.phi, self.constant)
        ma = MA(self.theta)

        return np.add(ar.generate_periods(periods),ma.generate_periods(periods))
        
class ARIMA:
    def __init__(self, phi, delta, theta, constant = 0.0):
        self.phi = phi
        self.delta = delta
        self.theta = theta
        self.constant = constant

    def generate_periods(self, periods):
        arma = ARMA(self.phi,self.theta, self.constant)

        return np.diff(arma.generate_periods(periods),self.delta)


class ARCH:
    def __init__(self, alpha):
        

        None

class GARCH:
    None

class HurstExponent:
    None