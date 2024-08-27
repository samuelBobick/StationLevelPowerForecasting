import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class STLARIMA:
    
    def __init__(self, lookahead=16, num_days_train = 30, alpha=2, period=96, p=1, d=0, q=2):
        """
        Args:
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            num_days_train (int, optional): How many days of past data to train model on. Defaults to 30.
            alpha (float, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
            period (int, optional): Number of readings per day. Defaults to 96.
            p (int, optional): ARIMA parameter p. Defaults to 1.
            d (int, optional): ARIMA parameter d. Defaults to 0.
            q (int, optional): ARIMA parameter q. Defaults to 2.
        """
        self.lookahead = lookahead
        self.num_days_train = num_days_train
        self.alpha = alpha
        self.period = period

        self.p = p
        self.d = d
        self.q = q

    def fit(self, train):
        """Train the model, save it as self.model

        Args:
            train (DataFrame): Training dataframe train with column "power"
        """
        power = train['power']
        stlf = STLForecast(power, ARIMA, model_kwargs={"order": (self.p, self.d, self.q)}, period=self.period)
        self.model = stlf.fit()

    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with column "power"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, list of predictions
        """
        power = test['power']

        all_forecasts = []
        all_actual = []
        for i in range(self.num_days_train * self.period, len(power), self.lookahead):
            train = power[i - self.num_days_train * self.period: i - self.lookahead]
            test = power[i-self.lookahead:i]

            stlf = STLForecast(train, ARIMA, model_kwargs={"order": (1, 0, 2)}, period=self.period)
            res = stlf.fit()
            forecast = res.forecast(self.lookahead)

            all_forecasts.extend(forecast)
            all_actual.extend(test)

        # Clip forecasts if they are negative
        all_forecasts = [x if x >= 0 else 0 for x in all_forecasts]

        rmse = np.sqrt(mean_squared_error(power[self.num_days_train * self.period : ], all_forecasts))
        weights = self.alpha ** (1 + np.sign(np.array(all_forecasts) - power[self.num_days_train * self.period : ]))
        wrmse = np.sqrt(mean_squared_error(forecast, all_forecasts, sample_weight=weights))

        return rmse, wrmse, all_forecasts
    