from typing import Literal

import pandas as pd
import xgboost as xgb

import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_ts_forecasting.regression_base import RegressionBaseModel


class XGBoost(RegressionBaseModel):

    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: bool = default_parameters.OPTIMIZE_LAGS,
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
        )
        self.alpha = alpha
        self.time_mode = time_mode

    @property
    def model_str_name(self):
        return "XGBoost" + ("_lagsOpti" if self.optimize_lags else "")

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
            "subsample": 0.8,  # fraction of training set to randomly sample for =
            # each tree (has a similar effect as dropout)
            "colsample_bytree": 0.8,
            "device": default_parameters.DEVICE,
        }
        num_round = 100
        bst = xgb.train(
            xgb_params, dtrain, num_round, evals=evallist, early_stopping_rounds=10
        )
        return bst

    def predict_model(self, model, X_test: pd.DataFrame):
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
