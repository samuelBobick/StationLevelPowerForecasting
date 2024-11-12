import pandas as pd
from slrp_ev_ts_forecasting.default_parameters import (
    NUMBER_OF_DAYS_FOR_PACF,
    TypeOptimizeLags,
)
from slrp_ev_ts_forecasting.pacf import get_pacf_values, get_threshold, sort_pacf_values


class Base:
    def __init__(self, lookahead, x_dim, optimize_lags: TypeOptimizeLags):
        self.lookahead = lookahead
        self.x_dim = x_dim
        self.optimize_lags = optimize_lags

    def get_top_pacf_values(
        self, data: pd.DataFrame, nb_of_days_for_pacf: int = NUMBER_OF_DAYS_FOR_PACF
    ) -> pd.Series:
        """Returns a list-like object with the top x_dim values of the Partial AutoCorrelation Function (PACF)."""
        # Define "number of steps to predict" given to the PACF function
        if self.optimize_lags == "long_opt":
            nb_of_steps_to_predict = 1
        elif self.optimize_lags == "short_opt":
            nb_of_steps_to_predict = self.lookahead
        else:
            raise ValueError("Optimize_lags should be 'short_opt' or 'long_opt'")

        pacf_df, interval = get_pacf_values(
            downsample_hours=1,
            data=data,
            nb_of_days_for_pacf=nb_of_days_for_pacf,
            nb_of_steps_to_predict=nb_of_steps_to_predict,
            return_confidence_interval=True,
        )

        number_of_lags_to_keep = self.x_dim
        pacf_top_values = sort_pacf_values(pacf_df, number_of_lags_to_keep)

        _, self.index_farthest_lag = get_threshold(
            pacf_df, number_of_lags_to_keep=number_of_lags_to_keep, interval=interval
        )
        print(f"Farthest lag: {self.index_farthest_lag / 96} days back")
        return pacf_top_values["PACF"]

    @property
    def seen_data(self):
        seen_data = getattr(self, "_seen_data", None)
        if seen_data is None:
            self._seen_data = pd.DataFrame(columns=["date"])
        return self._seen_data

    def update_seen_data(self, data: pd.DataFrame) -> None:
        # Concatenate the DataFrames
        concatenated_data = pd.concat([self.seen_data, data], ignore_index=True)

        # Identify duplicated dates
        duplicated_dates = concatenated_data[
            concatenated_data.duplicated(subset="date", keep=False)
        ]
        if not duplicated_dates.empty:
            print(
                f"Warning: {len(duplicated_dates)} duplicated dates found in the data. Dropping duplicates."
            )
            concatenated_data = concatenated_data.drop_duplicates(subset="date")

        # Assign the combined DataFrame to self._seen_data
        self._seen_data = concatenated_data

    def pad_with_seen_data(
        self, new_data_to_pad: pd.DataFrame, number_of_timesteps_to_pad: int
    ) -> pd.DataFrame:
        """Add at the beginning of the "new_data_to_pad" DataFrame the "number_of_timesteps_to_pad" that precede the given data.
        If the data is not available or some timesteps are missing, no padding is done.
        """
        first_date_of_data_to_pad = pd.to_datetime(
            new_data_to_pad.iloc[0]["date"], unit="s"
        )
        # Build padding index
        padding_index = pd.date_range(
            start=first_date_of_data_to_pad
            - pd.Timedelta(minutes=15) * number_of_timesteps_to_pad,
            periods=number_of_timesteps_to_pad,
            freq="15min",
        )

        seen_data = self.seen_data.copy()
        seen_data["date"] = pd.to_datetime(seen_data["date"], unit="s")
        seen_data["date"] = seen_data["date"].dt.round("5min")
        seen_data = seen_data.set_index("date")

        # Check if the padding index is in the model data
        if not padding_index.isin(seen_data.index).all():
            if not seen_data.empty:
                print(
                    "Warning: Some padding indexes are missing in the model data. Padding not done."
                )
            return new_data_to_pad

        # Get the padding data
        padding_data = seen_data.loc[padding_index]
        padding_data = padding_data.reset_index().rename(columns={"index": "date"})
        padding_data["date"] = padding_data["date"].astype("int64") // 10**9

        # Concatenate the padding data with the data to pad
        new_data_to_pad = pd.concat([padding_data, new_data_to_pad], ignore_index=True)

        return new_data_to_pad
