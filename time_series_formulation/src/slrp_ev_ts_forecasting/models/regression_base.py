from datetime import timedelta
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

        print("y_dates", y_dates)

        forecast_dates = y_dates.to_numpy().flatten()

        print(pd.to_datetime(forecast_dates, unit="s"))

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
        new_power_profile: bool = True,
        peak_prediction: bool = True,
    ):
        if new_power_profile:
            # if new_power_profile is True, we do session forecasting, and
            # we will look for the sessions in all of the windows,
            # so we need overlapping_windows to be True
            overlapping_windows = True

        if self.optimize_lags:
            input_width = self.index_farthest_lag
        else:
            input_width = self.x_dim

        if df is not None:
            df = df.copy()
            # We pad the data with input_width elements of the last seen data
            # so that we can predict the first elements of df
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

        label_cols_to_flatten = ["power"]
        if new_power_profile:
            label_cols_to_flatten += []

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
                label_cols_to_flatten=label_cols_to_flatten,
            )
        else:
            flat_inputs, flat_labels = W.flatten_dataset(
                window_data,
                cols_to_flatten=["power"],
                cols_keep_last_value=cols_keep_last_value,
                label_cols_to_flatten=label_cols_to_flatten,
            )

        if new_power_profile:
            # Look in sessions_df and round to the next 15-min interval
            sessions = pd.read_csv(
                "/Users/sam/Desktop/StationLevelPowerForecasting/data/Sessions3.csv"
            )
            sessions["startChargeTime"] = pd.to_datetime(sessions["startChargeTime"])
            # Round to the nearest 15-minute interval
            sessions["startChargeTime"] = sessions["startChargeTime"].dt.round("15min")

            power_df = pd.read_csv(
                "/Users/sam/Desktop/StationLevelPowerForecasting/data/src/slrp_ev_data/data/power_df_2008-2406_v241209_15min.csv"
            )
            power_df["recordTimestamp"] = pd.to_datetime(power_df["recordTimestamp"])

            x_dates, y_dates = W.flatten_dataset(
                window_data, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )

            if overlapping_windows:
                dates = pd.to_datetime(y_dates["date_0"], unit="s").dt.round("15min")
            else:
                dates = pd.concat(
                    [
                        pd.to_datetime(y_dates[f"date_{i}"], unit="s").dt.round("15min")
                        for i in range(96)
                    ],
                    axis=0,
                )
                dates = dates.sort_values()
                dates = dates.rename("date_0")
                print("dates shape for test", dates.shape)

            # flat_inputs = flat_inputs[dates.isin(sessions['startChargeTime'])]
            # flat_labels = flat_labels[dates.isin(sessions['startChargeTime'])]
            flat_labels = flat_labels.rename(
                columns={
                    col_name: f"{col_name}_label" for col_name in flat_labels.columns
                }
            )
            y_dates = y_dates.rename(
                columns={col_name: f"{col_name}_label" for col_name in y_dates.columns}
            )
            flat_inputs_dates = pd.concat(
                [flat_inputs, dates, flat_labels, y_dates], axis=1
            )
            print("flat_inputs_dates for test", flat_inputs_dates)

            merged_inputs_dates_sessions = sessions[
                ["startChargeTime", "dcosId"]
            ].merge(
                flat_inputs_dates,
                left_on="startChargeTime",
                right_on="date_0",
                how="inner",
            )
            print(
                "merged_inputs_dates_sessions shape for test",
                merged_inputs_dates_sessions.shape,
            )
            new_profiles = []
            for index, row in merged_inputs_dates_sessions.iterrows():
                dcosId = row["dcosId"]

                power_array = [None] * 96
                # Loop through columns and extract corresponding 'power' column based on the match
                for i in range(1, 9):
                    dcos_column = f"dcosId{i}"
                    power_column = f"power{i}"

                    if (power_df[dcos_column] == dcosId).any():
                        power_array = power_df[power_df[dcos_column] == dcosId][
                            power_column
                        ].to_numpy()
                        # print(power_df[power_df[dcos_column] == dcosId])

                        # Extract the two datetime objects
                        record_timestamp = power_df[
                            power_df[dcos_column] == dcosId
                        ].iloc[0]["recordTimestamp"]
                        start_charge_time = row["startChargeTime"]
                        time_difference = abs(record_timestamp - start_charge_time)

                        if time_difference >= timedelta(minutes=30):
                            print(
                                f"WARNING: Time difference exceeds 30 minutes: {time_difference} for session {dcosId}. startChargeTime: {start_charge_time}. recordTimestamp: {record_timestamp}"
                            )

                        if len(power_array) > 96:
                            print(f"WARNING: session {dcosId} power profile truncated")
                            power_array = power_array[:96]
                        elif len(power_array) < 96:
                            power_array = np.pad(
                                power_array, (0, 96 - len(power_array)), mode="constant"
                            )

                        break

                new_profiles.append(power_array)

            assert len(new_profiles) == len(
                merged_inputs_dates_sessions
            ), f"{len(new_profiles)} {len(merged_inputs_dates_sessions)}"
            new_profiles_df = pd.DataFrame(new_profiles)
            new_profiles_df.columns = [
                f"u_{i+1}" for i in range(96)
            ]  # TODO hardcoded 96 again
            merged_inputs_dates_sessions = pd.concat(
                [merged_inputs_dates_sessions, new_profiles_df], axis=1
            )

            merged_inputs_dates_sessions = merged_inputs_dates_sessions.dropna()
            merged_inputs_dates_sessions = merged_inputs_dates_sessions.reset_index(
                drop=True
            )

            flat_labels = merged_inputs_dates_sessions[
                [
                    col_name
                    for col_name in flat_labels.columns
                    if col_name.startswith("power")
                ]
            ]
            flat_labels = flat_labels.rename(
                columns={col_name: col_name[:-6] for col_name in flat_labels.columns}
            )
            flat_inputs = merged_inputs_dates_sessions[
                list(flat_inputs.columns) + [f"u_{i+1}" for i in range(96)]
            ]

        if return_y_date:
            if new_power_profile:
                y_dates = merged_inputs_dates_sessions[
                    [
                        col_name
                        for col_name in y_dates.columns
                        if col_name.startswith("date")
                    ]
                ]
                y_dates = y_dates.rename(
                    columns={col_name: col_name[:-6] for col_name in y_dates.columns}
                )
            else:
                x_dates, y_dates = W.flatten_dataset(
                    window_data,
                    cols_to_flatten=["date"],
                    label_cols_to_flatten=["date"],
                )
            return flat_inputs, flat_labels, y_dates
        else:
            return flat_inputs, flat_labels
