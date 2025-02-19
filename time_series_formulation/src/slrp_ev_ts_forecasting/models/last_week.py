import pandas as pd
from slrp_ev_data.feature_engineering import (
    convert_date_from_datetime_to_int,
    convert_date_from_int_to_datetime,
)
from slrp_ev_ts_forecasting.models.base import prepare_df_predictions


class LastWeek:

    def __init__(
        self,
        scaling_mode=None,
        scaling_parameters=None,
    ):
        """
        Args:
            scaling_mode (str, optional): UNUSED. Defaults to None.
            scaling_parameters (dict, optional): UNUSED. Defaults to None.
        """

    @property
    def model_str_name(self):
        return "LastWeek"

    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with column "power"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, list of predictions
        """
        test = test.copy()

        test["date"] = convert_date_from_int_to_datetime(test["date"])
        test_forecast = test.loc[
            test["date"].isin(test["date"] - pd.Timedelta(days=7))
        ].copy()
        test_forecast["date"] = test_forecast["date"] + pd.Timedelta(days=7)

        forecast = test_forecast["power"].to_numpy()
        real = test[test["date"].isin(test_forecast["date"])]["power"].to_numpy()
        forecast_dates = convert_date_from_datetime_to_int(test_forecast["date"])

        df_predictions = prepare_df_predictions(forecast, forecast_dates, real)

        return df_predictions

    def fit(self, train, val):
        """Fit method, not used in this model"""
        return
