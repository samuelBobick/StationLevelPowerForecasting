import numpy as np
import pandas as pd
from slrp_ev_data.feature_engineering import (
    convert_date_from_datetime_to_int,
    convert_date_from_int_to_datetime,
)
from slrp_ev_ts_forecasting.models.base import prepare_df_predictions


class SimilarDay:

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
        # TODO: we could average on a few days

    def predict(self, test: pd.DataFrame):
        """Given a pandas DataFrame test with a power column, it predicts the power for the next day \
            using the power of the last similar day (last workday/non-workday).

        Args:
            test (DataFrame): test set

        Returns:
            pd.DataFrame: Dataframe with the predictions
        """
        test = test.copy()
        test["date"] = convert_date_from_int_to_datetime(test["date"])
        previous_day_lookup = test.apply(
            self.apply_find_previous_workday, args=(test,), axis=1
        )  # type: ignore

        forecast = []
        for i in test.index:
            day_lookup = previous_day_lookup.loc[i]
            if pd.isna(day_lookup):
                forecast.append(None)
                continue

            try:
                forecast_power = test[test["date"] == day_lookup]["power"].iloc[0]
            except IndexError:
                forecast_power = None

            forecast.append(forecast_power)

        real = test.loc[:, "power"]
        forecast_dates = convert_date_from_datetime_to_int(
            test.loc[:, "date"]
        ).to_numpy()

        df_predictions = prepare_df_predictions(
            np.array(forecast), pd.Series(forecast_dates), real.to_numpy()
        )

        return df_predictions

    @property
    def model_str_name(self):
        return "SimilarDay"

    def fit(self, train, val):
        """Fit method, not used in this model"""
        return

    def find_past_similar_day(
        self, current_day_index, df: pd.DataFrame, _days_back: int = 1
    ) -> pd.Timestamp | None:
        """Recursive function to find the last similar day

        Args:
            current_day_index: index of the current day
            df (pd.DataFrame): dataframe with the data
            _days_back (int, optional): Argument for recursivity, leave to default. how many days back to look for a similar day. Defaults to 1.
        """
        day_back_date = df.loc[current_day_index]["date"] - pd.Timedelta(
            days=_days_back
        )
        is_workday_day_back = df.loc[df["date"] == day_back_date]["workday_0"]

        if is_workday_day_back.empty:
            # TODO: in case of missing data, we should look back an additional day
            return None
        else:
            is_workday_day_back = is_workday_day_back.iloc[0]

        if is_workday_day_back == df["workday_0"].loc[current_day_index]:
            return day_back_date
        else:
            return self.find_past_similar_day(current_day_index, df, _days_back + 1)

    def apply_find_previous_workday(
        self, row: pd.Series, df: pd.DataFrame
    ) -> pd.Timestamp | None:

        return self.find_past_similar_day(row.name, df)
