import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

class STLARIMA:
    
    def __init__(self, lookahead=16, alpha=0.2, period=96, p=1, d=0, q=2):
        self.lookahead = lookahead
        self.alpha = alpha

        self.p = p
        self.d = d
        self.q = q

        self.period = period


    def fit(self, train):
        power = train['power']
        stlf = STLForecast(power, ARIMA, model_kwargs={"order": (self.p, self.d, self.q)}, period=self.period)
        self.model = stlf.fit()

    def predict(self, test):
        power = test['power']

        all_forecasts = []
        all_actual = []
        for i in range(30 * self.period, len(power), self.lookahead):
            train = power[i - 30 * self.period: i - self.lookahead]
            test = power[i-self.lookahead:i]

            stlf = STLForecast(train, ARIMA, model_kwargs={"order": (1, 0, 2)}, period=self.period)
            res = stlf.fit()
            forecast = res.forecast(self.lookahead)

            all_forecasts.extend(forecast)
            all_actual.extend(test)


        all_forecasts = [x if x >= 0 else 0 for x in all_forecasts]

        mse = mean_squared_error(power[30 * self.period : ], all_forecasts)

        return mse
    
# data = pd.read_csv("/Users/sam/Desktop/StationLevelPowerForecasting/time_series_formulation/data.csv")
# train = data.loc[:71423]
# test = data.loc[71424:]

# model = STLARIMA()
# model.fit(train)
# print(model.predict(test))