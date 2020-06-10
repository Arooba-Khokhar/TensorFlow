# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:37:38 2018

@author: stech
"""

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
#from statsmodels.tsa.arima_model.ARIMAResults import forecast


from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from matplotlib import pyplot
 

 
series = read_csv('daily-minimum-temperatures-in-me.csv', header=0, parse_dates=[0], index_col=0,)# squeeze=True,)# date_parser=parser)
autocorrelation_plot(series)
pyplot.show()




#series = read_csv('daily-minimum-temperatures-in-me.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# fit model
model = ARIMA(series, order=(5,1,0))

#3650

train_data = series[0:int(3650 * 0.8)]
validation_data = series[int(3650 * 0.8):]

print(len(train_data),len(validation_data))

model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

forecast_array = model.forecast(steps=730)
output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())