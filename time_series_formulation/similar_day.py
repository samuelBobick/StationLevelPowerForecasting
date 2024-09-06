import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


class SimilarDay:

    def __init__(self, alpha=2, num_days=5, readings_per_day=96):
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
            apply_find_previous_workday, args=(test,), axis=1
        )
        first_index_that_can_be_forecasted = (
            test[previous_day_lookup.isna()].iloc[-1].name + 1
        )

        forecast = []
        for i in range(first_index_that_can_be_forecasted, test.index[-1] + 1):
            current_date = test.loc[i, "date"]

            forecast.append(
                test[test["date"] == (current_date - pd.Timedelta(days=previous_day_lookup.loc[i]))].iloc[0]["power"]  # type: ignore
            )

        real = test.loc[first_index_that_can_be_forecasted:, "power"]

        rmse = root_mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wrmse = root_mean_squared_error(forecast, real, sample_weight=weights)

        return rmse, wrmse, forecast

    def fit(self, train):
        return


def find_past_similar_day(
    current_day_index: int, df: pd.DataFrame, days_back: int
) -> int | None:
    is_worday_day_back = df["workday"].shift(96 * days_back).loc[current_day_index]
    if pd.isna(is_worday_day_back):
        return None
    elif is_worday_day_back == df["workday"].loc[current_day_index]:
        return days_back
    else:

        return find_past_similar_day(current_day_index, df, days_back + 1)


def apply_find_previous_workday(row: pd.Series, df: pd.DataFrame) -> int | None:

    return find_past_similar_day(row.name, df, days_back=1)
