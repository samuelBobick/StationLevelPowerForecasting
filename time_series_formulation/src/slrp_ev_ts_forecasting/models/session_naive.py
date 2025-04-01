from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class SessionNaive(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        scaling_mode: default_parameters.TypeScalingMode = default_parameters.SCALING_MODE,
        scaling_parameters: tuple | pd.DataFrame | None = None,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
        add_number_of_evses_available: bool = default_parameters.ADD_NUMBER_OF_EVSES_AVAILABLE,
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
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
            number_of_artificial_datasets=0,
            random_start_time=False,
            shuffle_power_profiles=False,
            random_power_profile_shapes=False,
            random_user_needs=False,
            random_choices=False,
            add_number_of_evses_available=add_number_of_evses_available,
        )
        self.alpha = alpha
        self.time_mode = time_mode
        if not session_based_mode:
            raise ValueError(
                "SessionNaive model can only be used for session prediction. Please set session_based_mode=True."
            )

    @property
    def model_str_name(self):
        return "SessionNaive" + self.model_str_name_suffix

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

    def predict_model(
        self,
        model,
        X_test: pd.DataFrame,
        mode: default_parameters.TypePeakPredictionMode = default_parameters.PEAK_PREDICTION_MODE,
    ):
        return X_test.filter(regex=r"u_")
