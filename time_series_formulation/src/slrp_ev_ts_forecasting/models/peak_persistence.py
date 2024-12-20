from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
)
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class PeakPersistence(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            alpha=alpha,
            time_mode=time_mode,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
        )
        self.alpha = alpha
        self.time_mode = time_mode
        if not peak_prediction:
            raise ValueError(
                "PeakPersistence model can only be used for peak prediction. Please set peak_prediction=True."
            )

    @property
    def model_str_name(self):
        return "PeakPersistence" + self.model_str_name_suffix

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_mask: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        val_mask: pd.Series,
    ):
        return None

    def predict_model(self, model, X_test: pd.DataFrame):
        # We have the information of the max before the current time
        # it's in the last 10 hours of the station power
        lookback_timesteps_for_peak = min(10 * 4, self.x_dim)
        max_before_current_time = (
            X_test[
                [
                    f"power_{i}"
                    for i in range(self.x_dim - lookback_timesteps_for_peak, self.x_dim)
                ]
            ]
            .max(axis=1)
            .iloc[0]
        )
        # After the current time, we don't have the exact load.
        # Therefore, we predict that the max after now is going to be
        # the load of the last known timestep + the max of the next session profile
        # We have to pu the max of the next session profile in the same scale as the load
        max_after_current_time = (
            X_test[f"power_{self.x_dim - 1}"].iloc[0]
            + X_test.filter(regex=r"u_").max(axis=1).iloc[0]
            * SINGLE_EVSE_NORMALIZATION_PARAM
            / 50_000
        )
        return max(max_before_current_time, max_after_current_time)
