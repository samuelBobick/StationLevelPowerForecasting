import time
from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
import xgboost as xgb
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class XGBoost(RegressionBaseModel):

    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        dropout: float = default_parameters.DROPOUT,
        max_depth: int = 4,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        scaling_mode: default_parameters.TypeScalingMode = default_parameters.SCALING_MODE,
        scaling_parameters: tuple | pd.DataFrame | None = None,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
        number_of_artificial_datasets: int = default_parameters.NUMBER_OF_ARTIFICIAL_DATASETS,
        random_start_time: bool = default_parameters.RANDOM_START_TIME,
        shuffle_power_profiles: bool = default_parameters.SHUFFLE_POWER_PROFILES,
        random_power_profile_shapes: bool = default_parameters.RANDOM_POWER_PROFILE_SHAPES,
        random_user_needs: bool = default_parameters.RANDOM_USER_NEEDS,
        random_choices: bool = default_parameters.RANDOM_CHOICES,
        add_number_of_evses_available: bool = default_parameters.ADD_NUMBER_OF_EVSES_AVAILABLE,
    ):
        """_summary_

        Args:
            x_dim: How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead: How many timesteps ahead we want to predict. Defaults to 16.
            alpha: Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
            time_mode:
            optimize_lags:
            dropout:
            max_depth: Does not seem to have a huge impact on the results, but has a big impact
                on the computation time. Default in XGBoost module is 6.
            get_val_data_from_shuffled_train:
            scaling_mode:
            scaling_parameters:
            session_based_mode:
            peak_prediction:
            add_number_of_sessions:
            add_fraction_of_regular_sessions:
            use_all_active_sessions:
            number_of_artificial_datasets:
            random_start_time:
            shuffle_power_profiles:
            random_power_profile_shapes:
            random_user_needs:
            random_choices:
            add_number_of_evses_available:

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
            number_of_artificial_datasets=number_of_artificial_datasets,
            random_start_time=random_start_time,
            shuffle_power_profiles=shuffle_power_profiles,
            random_power_profile_shapes=random_power_profile_shapes,
            random_user_needs=random_user_needs,
            random_choices=random_choices,
            add_number_of_evses_available=add_number_of_evses_available,
        )
        self.alpha = alpha
        self.time_mode = time_mode
        self.dropout = dropout
        self.max_depth = max_depth

    @property
    def model_str_name(self):
        return (
            "XGBoost"
            + f"_dropout{self.dropout}"
            + f"_max_depth{self.max_depth}"
            + self.model_str_name_suffix
        )

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_mask: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        val_mask: pd.Series,
    ):
        dtrain = xgb.DMatrix(
            X_train[train_mask].drop(
                self.cols_to_drop_for_model,
                axis=1,
            ),
            label=y_train[train_mask],
        )
        dval = xgb.DMatrix(
            X_val[val_mask].drop(
                self.cols_to_drop_for_model,
                axis=1,
            ),
            label=y_val[val_mask],
        )
        evallist = [
            (dtrain, "train"),
            (dval, "eval"),
        ]

        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,  # default is 6
            "eta": 0.1,  # learning rate, default 0.3
            "subsample": 1
            - self.dropout,  # fraction of training set to randomly sample for =
            # each tree (has a similar effect as dropout)
            "colsample_bytree": 1 - self.dropout,  # fraction of features to consider
            "device": default_parameters.DEVICE,
            "max_bin": 128,  # default is 256
            "seed": int(time.time()),  # add random seed, otherwise default is 0
        }
        num_round = 100
        bst = xgb.train(
            xgb_params, dtrain, num_round, evals=evallist, early_stopping_rounds=10
        )
        return bst

    def predict_model(self, model, X_test: pd.DataFrame):
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
