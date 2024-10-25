import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_ts_forecasting.compute_losses import compute_losses


class LastWeek:

    def __init__(self, alpha=default_parameters.ALPHA, readings_per_day=96):
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

        losses = compute_losses(forecast, real, self.alpha)

        return losses, forecast, forecast_dates

    def fit(self, train):
        return
