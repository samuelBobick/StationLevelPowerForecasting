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
