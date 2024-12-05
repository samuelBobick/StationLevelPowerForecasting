from typing import Literal

import numpy as np
import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from sklearn import linear_model
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class LinearModel(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
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
        )
        self.alpha = alpha
        self.time_mode = time_mode
        

    @property
    def model_str_name(self):
        return (
            f"LinearModel"
            + ("_lagsOpti" if self.optimize_lags else "")
            + ("Short" if self.optimize_lags == "short_opt" else "")
            + ("Long" if self.optimize_lags == "long_opt" else "")
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

        X_input = X_train[train_mask].drop(
            self.cols_to_drop_for_model,
            # [col for col in X_train.columns if not col.startswith("power")],
            axis=1,
        )
        y_input = y_train[train_mask]

        lm = linear_model.LinearRegression()
        lm.fit(X_input, y_input)
        return lm

    def predict_model(self, model, X_test: pd.DataFrame):
        return model.predict(X_test)

