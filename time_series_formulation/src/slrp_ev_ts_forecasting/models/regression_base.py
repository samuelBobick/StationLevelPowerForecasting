from typing import Literal

import numpy as np
import pandas as pd
from slrp_ev_ts_forecasting.compute_losses import compute_losses
from slrp_ev_ts_forecasting.default_parameters import TypeOptimizeLags
from slrp_ev_ts_forecasting.models.base import Base
from tqdm import tqdm


class RegressionBaseModel(Base):

    def __init__(
        self,
        x_dim: int,
        lookahead: int,
        alpha: float,
        time_mode: Literal["window", "cyclical"],
        optimize_lags: TypeOptimizeLags,
        get_val_data_from_shuffled_train: bool,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. \
                Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (int, optional): Underpredictions are penalized alpha times more than \
                overpredictions for weighted error metric. Defaults to 2.
            get_val_data_from_shuffled_train (bool): Whether to get the \
                validation data from the shuffled train data. This can help \
                improving the algorithm's performance since there will more \
                recent data in the training set (otherwise, the most recent data \
                is in the val and test sets)
        """
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
        )

        self.lookahead = lookahead
        self.x_dim = x_dim

        self.optimize_lags = optimize_lags

        self.alpha = alpha
        self.time_mode = time_mode
        if self.time_mode == "window":
            self.cols_to_drop_for_model = [
                "time_window",
                "workday",
            ]
        elif self.time_mode == "cyclical":
            self.cols_to_drop_for_model = [
                "workday",
                # "Year sin",
                # "Year cos",
            ]

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        if self.optimize_lags:
            self.pacf_top_values = self.get_top_pacf_values(train)

        X_train, y_train = self.get_X_y(train, data_type="train", overlapping_windows=True)  # type: ignore
        self.update_seen_data(train)
        X_val, y_val = self.get_X_y(val, data_type="val", overlapping_windows=False)  # type: ignore
        if val is not None:
            self.update_seen_data(val)

        self.models = {}
        if self.time_mode == "window":
            for t_w in range(6):
                for w in [0, 1]:
                    train_mask = (X_train["time_window"] == t_w) & (
                        X_train["workday"] == w
                    )
                    val_mask = (X_val["time_window"] == t_w) & (X_val["workday"] == w)

                    self.models[(t_w, w)] = self.fit_model(
                        X_train, y_train, train_mask, X_val, y_val, val_mask
                    )
        elif self.time_mode == "cyclical":
            for w in [0, 1]:
                train_mask = X_train["workday"] == w
                val_mask = X_val["workday"] == w
                self.models[w] = self.fit_model(
                    X_train, y_train, train_mask, X_val, y_val, val_mask
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
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def predict(self, test):
        """
        Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        X_test, y_test, y_dates = self.get_X_y(test, data_type="test", return_y_date=True)  # type: ignore

        forecasts = []

        for index, row in tqdm(X_test.iterrows(), desc="Predicting", total=len(X_test)):
            input = pd.DataFrame(
                [row.drop(self.cols_to_drop_for_model)]
            )  # .to_numpy().reshape(1, -1)
            if self.time_mode == "window":
                forecasts.append(
                    self.predict_model(
                        self.models[(row["time_window"], row["workday"])], input
                    )
                )
            elif self.time_mode == "cyclical":
                forecasts.append(self.predict_model(self.models[row["workday"]], input))

        forecast = np.array([f[0] for f in forecasts]).flatten()
        real = y_test.to_numpy().flatten()

        losses = compute_losses(forecast, real, self.alpha)

        forecast_dates = y_dates.to_numpy().flatten()

        return losses, forecast, forecast_dates

    def predict_model(self, model, X_test: pd.DataFrame):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def get_X_y(
        self,
        df: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ):
        if self.optimize_lags:
            input_width = self.index_farthest_lag
        else:
            input_width = self.x_dim

        if df is not None:
            df = df.copy()
            df_padded = self.pad_with_seen_data(df, input_width)
        else:
            df_padded = None

        W, window_data = self.get_window_data(
            df_padded, input_width, self.lookahead, overlapping_windows, data_type
        )

        cols_keep_last_value = ["workday"]
        if self.time_mode == "cyclical":
            cols_keep_last_value += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ]
        elif self.time_mode == "window":
            cols_keep_last_value += ["time_window"]

        if self.optimize_lags:
            flat_inputs, flat_labels = W.flatten_dataset(
                window_data,
                cols_keep_last_value=cols_keep_last_value,
                cols_keep_some_values=[
                    {
                        "col_name": "power",
                        "indexes_to_keep": input_width
                        - self.pacf_top_values.index.to_numpy(),
                    }
                ],
                label_cols_to_flatten=["power"],
            )
        else:
            flat_inputs, flat_labels = W.flatten_dataset(
                window_data,
                cols_to_flatten=["power"],
                cols_keep_last_value=cols_keep_last_value,
                label_cols_to_flatten=["power"],
            )
        mask_nan = flat_inputs.isna().any(axis=1) | flat_labels.isna().any(axis=1)
        flat_inputs = flat_inputs[~mask_nan]
        flat_labels = flat_labels[~mask_nan]
        print(flat_inputs.shape, flat_labels.shape)

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                window_data, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            y_dates = y_dates[~mask_nan]
            return flat_inputs, flat_labels, y_dates
        else:
            return flat_inputs, flat_labels
