from typing import Literal

import numpy as np
import pandas as pd
from slrp_ev_data.window_generator import WindowGenerator
from tqdm import tqdm

from slrp_ev_ts_forecasting.base import Base
from slrp_ev_ts_forecasting.compute_losses import compute_losses
from slrp_ev_ts_forecasting.default_parameters import TypeOptimizeLags


class RegressionBaseModel(Base):

    def __init__(
        self,
        x_dim: int,
        lookahead: int,
        alpha: float,
        time_mode: Literal["window", "cyclical"],
        optimize_lags: TypeOptimizeLags,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        super().__init__(x_dim=x_dim, lookahead=lookahead, optimize_lags=optimize_lags)

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

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        if self.optimize_lags:
            self.pacf_top_values = self.get_top_pacf_values(train)

        X_train, y_train = self.get_X_y(train, overlapping_windows=True)  # type: ignore
        self.update_seen_data(train)
        X_val, y_val = self.get_X_y(val, overlapping_windows=False)  # type: ignore
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
        X_test, y_test, y_dates = self.get_X_y(test, return_y_date=True)  # type: ignore

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
        df,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ):
        df = df.copy()

        if self.optimize_lags:
            input_width = self.index_farthest_lag
        else:
            input_width = self.x_dim

        df_padded = self.pad_with_seen_data(df, input_width)

        W = WindowGenerator(
            input_width=input_width,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df_padded,
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

        if self.optimize_lags:
            flat_inputs, flat_labels = W.flatten_dataset(
                W.train,
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

    @property
    def seen_data(self):
        seen_data = getattr(self, "_seen_data", None)
        if seen_data is None:
            self._seen_data = pd.DataFrame(columns=["date"])
        return self._seen_data

    def update_seen_data(self, data: pd.DataFrame) -> None:
        # Concatenate the DataFrames
        concatenated_data = pd.concat([self.seen_data, data], ignore_index=True)

        # Identify duplicated dates
        duplicated_dates = concatenated_data[
            concatenated_data.duplicated(subset="date", keep=False)
        ]
        if not duplicated_dates.empty:
            print(
                f"Warning: {len(duplicated_dates)} duplicated dates found in the data. Dropping duplicates."
            )
            concatenated_data = concatenated_data.drop_duplicates(subset="date")

        # Assign the combined DataFrame to self._seen_data
        self._seen_data = concatenated_data

    def pad_with_seen_data(
        self, new_data_to_pad: pd.DataFrame, number_of_timesteps_to_pad: int
    ) -> pd.DataFrame:
        """Add at the beginning of the "new_data_to_pad" DataFrame the "number_of_timesteps_to_pad" that precede the given data.
        If the data is not available or some timesteps are missing, no padding is done.
        """
        first_date_of_data_to_pad = pd.to_datetime(
            new_data_to_pad.iloc[0]["date"], unit="s"
        )
        # Build padding index
        padding_index = pd.date_range(
            start=first_date_of_data_to_pad
            - pd.Timedelta(minutes=15) * number_of_timesteps_to_pad,
            periods=number_of_timesteps_to_pad,
            freq="15min",
        )

        seen_data = self.seen_data.copy()
        seen_data["date"] = pd.to_datetime(seen_data["date"], unit="s")
        seen_data["date"] = seen_data["date"].dt.round("5min")
        seen_data = seen_data.set_index("date")

        # Check if the padding index is in the model data
        if not padding_index.isin(seen_data.index).all():
            if not seen_data.empty:
                print(
                    "Warning: Some padding indexes are missing in the model data. Padding not done."
                )
            return new_data_to_pad

        # Get the padding data
        padding_data = seen_data.loc[padding_index]
        padding_data = padding_data.reset_index().rename(columns={"index": "date"})
        padding_data["date"] = padding_data["date"].astype("int64") // 10**9

        # Concatenate the padding data with the data to pad
        new_data_to_pad = pd.concat([padding_data, new_data_to_pad], ignore_index=True)

        return new_data_to_pad
