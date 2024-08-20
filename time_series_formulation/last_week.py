import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



class LastWeek:

    def __init__(self, alpha=0.2, readings_per_day=96):
        
        self.alpha = alpha
        self.readings_per_day = readings_per_day

    def predict(self, test):
        power = test['power'].to_numpy()

        forecast = power[:-self.readings_per_day]
        real = power[self.readings_per_day:]

        mse = mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wmse = np.sqrt(mean_squared_error(forecast, real, sample_weight=weights))

        return mse, wmse, forecast
    

