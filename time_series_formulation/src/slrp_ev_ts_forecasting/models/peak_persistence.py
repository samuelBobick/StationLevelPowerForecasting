from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
    retrieve_train_min_and_max,
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
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
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
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
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
        max_after_current_time = X_test.filter(regex=r"u_").max(axis=1).iloc[0]

        scale_factor = 8
        if not self.use_all_active_sessions:
            max_after_current_time += X_test[f"power_{self.x_dim - 1}"].iloc[0]
            scale_factor = 1

        _, train_max = retrieve_train_min_and_max("slrp-ev_new")
        max_after_current_time = (
            max_after_current_time
            * SINGLE_EVSE_NORMALIZATION_PARAM
            * scale_factor
            / train_max.iloc[0]
        )
        return max(max_before_current_time, max_after_current_time)
