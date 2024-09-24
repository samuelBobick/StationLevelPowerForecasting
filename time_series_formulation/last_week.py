import numpy as np
from sklearn.metrics import root_mean_squared_error


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
        power = test["power"].to_numpy()

        forecast = power[: -self.readings_per_day * 7]
        real = power[self.readings_per_day * 7 :]
        forecast_dates = test.iloc[self.readings_per_day * 7 :]["date"].to_numpy()

        rmse = root_mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        rwmse = root_mean_squared_error(forecast, real, sample_weight=weights)

        return rmse, rwmse, forecast, forecast_dates

    def fit(self, train):
        return
