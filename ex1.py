# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller

temp = pd.read_csv('daily-minimum-temperatures-in-me.csv', parse_dates=True, header=0, index_col=0)
series = temp["#Melbourne"]
print(series.values)
index = (pd.date_range('19810101', periods=3650,tz='US/Eastern')).values
print((pd.date_range('19810101', periods=3650,tz='US/Eastern')).values)

ans = pd.Series(series.values, index)

print(len(index))

plt.plot(ans)

df = pd.read_csv("AirPassengers.csv", header=0)
timeseries = df["#Passengers"]
plt.plot(timeseries)


def stationary_check(timeseries):
    rol_mean = pd.rolling_mean(timeseries, window=12)
    rol_std = pd.rolling_std(timeseries, window=12)
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Moving mean & std')
    plt.show(block=False)
stationary_check(timeseries)


log_timeseries = np.log(timeseries)#take the log to reduce the values
expwighted_avg = pd.ewma(log_timeseries, halflife=12)
difference = log_timeseries - expwighted_avg
stationary_check (difference)
