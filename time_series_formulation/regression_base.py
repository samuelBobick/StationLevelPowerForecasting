from typing import Literal

import default_parameters
import numpy as np
import pandas as pd
from asymmetric_loss import asymmetric_rmse
from sklearn.metrics import root_mean_squared_error  # type: ignore
from slrp_ev_data.window_generator import WindowGenerator


class RegressionBaseModel:

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        self.lookahead = lookahead
        self.x_dim = x_dim

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
                "Year sin",
                "Year cos",
            ]

    def fit(self, train: pd.DataFrame):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        X_train, y_train = self.get_X_y(train, overlapping_windows=True)  # type: ignore

        self.models = {}
        if self.time_mode == "window":
            for t_w in range(6):
                for w in [0, 1]:
                    mask = (X_train["time_window"] == t_w) & (X_train["workday"] == w)

                    self.models[(t_w, w)] = self.fit_model(X_train, y_train, mask)
        elif self.time_mode == "cyclical":
            for w in [0, 1]:
                mask = X_train["workday"] == w
                self.models[w] = self.fit_model(X_train, y_train, mask)

    def fit_model(self, X_train, y_train, data_mask):
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
        X_test, y_test, y_dates = self.get_X_y(test, return_y_date=True)  # type: ignore

        rmses = []
        rwmses = []

        forecasts = []

        for index, row in X_test.iterrows():
            input = row.drop(self.cols_to_drop_for_model).to_numpy().reshape(1, -1)
            if self.time_mode == "window":
                forecasts.append(
                    self.models[(row["time_window"], row["workday"])].predict(input)
                )
            elif self.time_mode == "cyclical":
                forecasts.append(self.models[row["workday"]].predict(input))

        forecast = np.array([f[0] for f in forecasts]).flatten()
        real = y_test.to_numpy().flatten()

        rmse = root_mean_squared_error(forecast, real)

        rwmse = asymmetric_rmse(self.alpha, forecast, real)

        rmses.append(rmse)
        rwmses.append(rwmse)

        forecast_dates = y_dates.to_numpy().flatten()

        return rmse, rwmse, forecast, forecast_dates

    def get_X_y(
        self,
        df,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ):
        df = df.copy()

        W = WindowGenerator(
            input_width=self.x_dim,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
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

        flat_inputs, flat_labels = W.flatten_dataset(
            W.train,
            cols_to_flatten=["power"],
            cols_keep_last_value=cols_keep_last_value,
            label_cols_to_flatten=["power"],
        )
        print(flat_inputs.shape, flat_labels.shape)

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                W.train, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            return flat_inputs, flat_labels, y_dates
        else:
            return flat_inputs, flat_labels
