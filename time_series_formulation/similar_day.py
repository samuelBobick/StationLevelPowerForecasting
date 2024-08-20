import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class SimilarDay:

    def __init__(self, alpha=0.2, num_days = 5, readings_per_day=96):
        self.alpha = alpha
        self.num_days = num_days
        self.readings_per_day = readings_per_day

    def predict(self, test):
        power = test['power']

        forecast = []
        for i in range(self.num_days * self.readings_per_day * 7, len(power)):
            forecast.append(np.mean(power[i - self.readings_per_day*self.num_days : i : self.readings_per_day]))

        real = power[self.num_days * self.readings_per_day * 7:]

        mse = mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wmse = mean_squared_error(forecast, real, sample_weight=weights)

        return mse, wmse
