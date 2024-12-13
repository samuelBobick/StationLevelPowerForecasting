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
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            n_neighbors (int, optional): K in the KNN algorithm. Defaults to 10.
            percentile (int, optional): What percentile of the KNN we take. Defaults to 90.
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
        self.dropout = dropout

    @property
    def model_str_name(self):
        return "XGBoost" + f"_dropout{self.dropout}" + self.model_str_name_suffix

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
            # "max_depth": 6,
            "eta": 0.1,  # learning rate, default 0.3
            "subsample": 1
            - self.dropout,  # fraction of training set to randomly sample for =
            # each tree (has a similar effect as dropout)
            "colsample_bytree": 1,  # - self.dropout,  # fraction of features to consider
            "device": default_parameters.DEVICE,
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
