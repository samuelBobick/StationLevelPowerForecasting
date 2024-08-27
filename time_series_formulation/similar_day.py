import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class SimilarDay:

    def __init__(self, alpha=2, num_days = 5, readings_per_day=96):
        """
        Args:
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions. Defaults to 2.
            num_days (int, optional): how many past days of the same data/time to average. Defaults to 5.
            readings_per_day (int, optional): Defaults to 96.
        """
        self.alpha = alpha
        self.num_days = num_days
        self.readings_per_day = readings_per_day

    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): 

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, list of predictions
        """
        power = test['power']

        forecast = []
        for i in range(self.num_days * self.readings_per_day * 7, len(power)):
            forecast.append(np.mean(power[i - self.readings_per_day*self.num_days : i : self.readings_per_day]))

        real = power[self.num_days * self.readings_per_day * 7:]

        rmse = mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wrmse = np.sqrt(mean_squared_error(forecast, real, sample_weight=weights))

        return rmse, wrmse, forecast
