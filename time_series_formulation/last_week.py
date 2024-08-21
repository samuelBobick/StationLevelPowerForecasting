import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



class LastWeek:

    def __init__(self, alpha=2, readings_per_day=96):
        """
        Args:
            alpha (float, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
            readings_per_day (int, optional): Defaults to 96.
        """
        
        self.alpha = alpha
        self.readings_per_day = readings_per_day

    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with column "power"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, list of predictions
        """
        power = test['power'].to_numpy()

        forecast = power[:-self.readings_per_day]
        real = power[self.readings_per_day:]

        mse = mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wmse = np.sqrt(mean_squared_error(forecast, real, sample_weight=weights))

        return mse, wmse, forecast
    

