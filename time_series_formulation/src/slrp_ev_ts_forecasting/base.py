import pandas as pd

from slrp_ev_ts_forecasting.default_parameters import TypeOptimizeLags
from slrp_ev_ts_forecasting.pacf import get_pacf_values, get_threshold


class Base:
    def __init__(self, lookahead, x_dim, optimize_lags: TypeOptimizeLags):
        self.lookahead = lookahead
        self.x_dim = x_dim
        self.optimize_lags = optimize_lags

    def get_top_pacf_values(
        self, data: pd.DataFrame, nb_of_days_for_pacf: int = 40
    ) -> pd.Series:
        """Returns a list-like object with the top x_dim values of the Partial AutoCorrelation Function (PACF)."""
        # Define "number of steps to predict" given to the PACF function
        if self.optimize_lags == "short_opt":
            nb_of_steps_to_predict = 1
        elif self.optimize_lags == "long_opt":
            nb_of_steps_to_predict = self.lookahead
        else:
            raise ValueError("Optimize_lags should be 'short_opt' or 'long_opt'")

        pacf_df = get_pacf_values(
            downsample_hours=1,
            data=data,
            nb_of_days_for_pacf=nb_of_days_for_pacf,
            nb_of_steps_to_predict=nb_of_steps_to_predict,
        )
        # remove first value (autocorrelation with itself)
        pacf_df = pacf_df.iloc[1:]

        number_of_lags_to_keep = self.x_dim
        pacf_top_values = (
            pacf_df["PACF"].sort_values(ascending=False).iloc[:number_of_lags_to_keep]
        )

        _, self.index_farthest_lag = get_threshold(
            pacf_df, number_of_lags_to_keep=number_of_lags_to_keep
        )
        print(f"Farthest lag: {self.index_farthest_lag / 96} days back")
        return pacf_top_values
