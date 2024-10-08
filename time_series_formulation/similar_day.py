import default_parameters
import pandas as pd
from asymmetric_loss import asymmetric_rmse
from sklearn.metrics import root_mean_squared_error  # type: ignore


class SimilarDay:

    def __init__(self, alpha=default_parameters.ALPHA, num_days=5, readings_per_day=96):
        """
        Args:
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions. Defaults to 2.
            num_days (int, optional): how many past days of the same data/time to average. Defaults to 5.
            readings_per_day (int, optional): Defaults to 96.
        """
        self.alpha = alpha
        self.num_days = num_days
        self.readings_per_day = readings_per_day

    def predict(self, test: pd.DataFrame):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame):

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, list of predictions
        """
        previous_day_lookup = test.apply(
            self.apply_find_previous_workday, args=(test,), axis=1
        )  # type: ignore
        first_index_that_can_be_forecasted = (
            test[previous_day_lookup.isna()].iloc[-1].name + 1
        )

        forecast = []
        seconds_in_day = 24 * 60 * 60
        for i in range(first_index_that_can_be_forecasted, test.index[-1] + 1):
            current_date = test.loc[i, "date"]

            forecast.append(
                test[
                    test["date"]
                    == (current_date - seconds_in_day * previous_day_lookup.loc[i])
                ].iloc[0]["power"]
            )

        real = test.loc[first_index_that_can_be_forecasted:, "power"]
        forecast_dates = test.loc[
            first_index_that_can_be_forecasted:, "date"
        ].to_numpy()

        rmse = root_mean_squared_error(forecast, real)

        rwmse = asymmetric_rmse(self.alpha, forecast, real)

        return rmse, rwmse, forecast, forecast_dates

    def fit(self, train):
        return

    def find_past_similar_day(
        self, current_day_index: int, df: pd.DataFrame, days_back: int
    ) -> int | None:
        is_workday_day_back = (
            df["workday"]
            .shift(self.readings_per_day * days_back)
            .loc[current_day_index]
        )
        if pd.isna(is_workday_day_back):
            return None
        elif is_workday_day_back == df["workday"].loc[current_day_index]:
            return days_back
        else:
            return self.find_past_similar_day(current_day_index, df, days_back + 1)

    def apply_find_previous_workday(
        self, row: pd.Series, df: pd.DataFrame
    ) -> int | None:

        return self.find_past_similar_day(row.name, df, days_back=1)
