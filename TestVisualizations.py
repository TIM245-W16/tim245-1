import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader as dl
import Helpers as hlp
import arch
import statsmodels.api as sm
from scipy.signal import detrend
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


for cur in dl.valid_currencies():
    print 'Evaluating {}'.format(cur)

    c_train, c_test = dl.load_test_train(cur)
    
    train_volatility = hlp.rolling_standard_dev(c_train.values)
    test_volatility = hlp.rolling_standard_dev(c_test.values)

    garch_model = arch.arch_model(train_volatility).fit()

    params = garch_model.params['omega'],\
                garch_model.params['alpha[1]'],\
                garch_model.params['beta[1]']
        
    forecaster = hlp.GarchForecaster(params)

    garch_predict = pd.Series(test_volatility,dtype=float)\
                        .apply(lambda x: forecaster.forecast(x))\
                        .shift(1)
        

    dFrame = pd.DataFrame(test_volatility, columns=['Actual'])
    dFrame['Predict'] = garch_predict
    dFrame.plot(title=cur)
        
        
    y = dFrame.loc[1:,'Actual']
    yhat = dFrame.loc[1:,'Predict']
    mfe = hlp.mean_forecast_err(y,yhat)
    mae = hlp.mean_absolute_err(y,yhat)
        
    print 'MFE={}\nMAE={}\n\n'.format(mfe,mae*100)

plt.show()
        



#http://www.quantatrisk.com/2014/10/23/garch11-model-in-python/
#https://onlinecourses.science.psu.edu/stat510/node/62
#http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html
#http://cims.nyu.edu/~almgren/timeseries/Vol_Forecast1.pdf