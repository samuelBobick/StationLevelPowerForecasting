from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from slrp_ev_data.feature_engineering import convert_date_from_int_to_datetime
from slrp_ev_data.read_new_slrpev_data import read_new_slrpev_data
from slrp_ev_ts_forecasting.compute_losses import Losses, compute_losses
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
        session_based_mode: bool,
        peak_prediction: bool,
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
        self.session_based_mode = session_based_mode
        self.peak_prediction = peak_prediction

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

    def predict(self, test) -> tuple[Losses, pd.DataFrame]:
        """
        Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (Losses, DataFrame): Losses object, DataFrame of predictions with \
                "date" and multiple "power_x" columns for the predictions
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

        forecast = np.array(forecasts).squeeze().flatten()
        real = y_test.to_numpy().flatten()
        losses = compute_losses(forecast, real, self.alpha)

        # TODO: Put following loop in a function in Base class + use it in all the
        # other predict functions
        predictions_array = np.stack(
            (y_dates.to_numpy(), np.array(forecasts).squeeze()), axis=-1
        )
        # Reshape to add a 3rd dimension in case we predict only the peak
        if len(predictions_array.shape) == 2:
            predictions_array = np.expand_dims(predictions_array, axis=1)

        df_predictions = pd.DataFrame(columns=["date"])
        for i in range(predictions_array.shape[0]):
            df_single_prediction = pd.DataFrame(
                {
                    "date": predictions_array[i, :, 0],
                    "power_0": predictions_array[i, :, 1],
                }
            )

            if df_single_prediction["date"].isin(df_predictions["date"]).any():
                # if we already have prediction data for these timesteps, we need to iterate
                # over the other power_x columns to find the first one that doesn't have data yet
                # if they all have data, we create a new column
                df_predictions_these_dates = df_predictions[
                    df_predictions["date"].isin(df_single_prediction["date"])
                ]
                next_power_column_number = len(df_predictions.columns) - 1
                # by default we add the data to a new column
                df_single_prediction = df_single_prediction.rename(
                    columns={"power_0": f"power_{next_power_column_number}"}
                )
                # If possible, we add it to an existing column
                for j in range(0, next_power_column_number):
                    if df_predictions_these_dates[f"power_{j}"].isna().all():
                        df_single_prediction = df_single_prediction.rename(
                            columns={f"power_{next_power_column_number}": f"power_{j}"}
                        )
                        break
                df_predictions = df_predictions.merge(df_single_prediction, how="outer")
            else:
                if df_predictions.empty:
                    # in the initial case, we have a pandas warning with the
                    # concat operation if the dataframe is empty, we solve it like so
                    df_predictions = df_single_prediction
                else:
                    df_predictions = pd.concat(
                        [df_predictions, df_single_prediction], ignore_index=True
                    )
        return losses, df_predictions

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
        if self.peak_prediction and not self.session_based_mode:
            raise ValueError(
                "self.peak_prediction can only be True if self.session_based_mode is True"
            )
        if self.session_based_mode:
            # if self.session_based_mode is True, we do session forecasting, and
            # we will look for the sessions in all of the windows,
            # so we need overlapping_windows to be True
            overlapping_windows = True

        if self.optimize_lags:
            # TODO: Fix optimize lags for missing data
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
        if self.session_based_mode:
            label_cols_to_flatten += ["date"]

        input_cols_to_flatten = ["power"]
        if self.peak_prediction:
            input_cols_to_flatten += ["date"]

        if self.optimize_lags:
            flat_inputs, flat_labels = W.flatten_dataset(
                window_data,
                cols_keep_last_value=cols_keep_last_value,
                cols_keep_some_values=[
                    {
                        "col_name": col_to_flatten,
                        "indexes_to_keep": input_width
                        - self.pacf_top_values.index.to_numpy(),
                    }
                    for col_to_flatten in input_cols_to_flatten
                ],
                label_cols_to_flatten=label_cols_to_flatten,
            )
        else:
            flat_inputs, flat_labels = W.flatten_dataset(
                window_data,
                cols_to_flatten=input_cols_to_flatten,
                cols_keep_last_value=cols_keep_last_value,
                label_cols_to_flatten=label_cols_to_flatten,
            )

        if self.session_based_mode:
            flat_inputs, flat_labels, merged_inputs_dates_sessions = (
                self.transform_X_y_for_session_based_mode(flat_inputs, flat_labels)
            )

        if return_y_date:
            if self.session_based_mode:
                if self.peak_prediction:
                    y_dates = merged_inputs_dates_sessions["date_0_label"]
                else:
                    y_dates = merged_inputs_dates_sessions.filter(
                        regex=r"date_(\d+)_label"
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

    def transform_X_y_for_session_based_mode(
        self,
        flat_inputs: pd.DataFrame,
        flat_labels: pd.DataFrame,
        max_timesteps_per_session: int = 96,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: Also add the number of current active sessions as a feature? Information is
        # maybe already in the "load until the start of this session"
        # TODO: Change the loss function in the model, to better predict peaks

        power_df = read_new_slrpev_data(keep_all_columns=True)

        start_dates_prediction_window = (
            pd.to_datetime(flat_labels["date_0"], unit="s")
            .dt.round("15min")
            .rename("start_date_prediction_window")
        )

        # We need to rename the label columns because "power_x" is already a column in flat_inputs
        flat_labels = flat_labels.rename(
            columns={col_name: f"{col_name}_label" for col_name in flat_labels.columns}
        )

        assert (
            flat_inputs.shape[0]
            == start_dates_prediction_window.shape[0]
            == flat_labels.shape[0]
        ), f"Concatenation of arrays of different sizes: {flat_inputs.shape[0]} {start_dates_prediction_window.shape[0]} {flat_labels.shape[0]}"
        flat_inputs_dates = pd.concat(
            [flat_inputs, start_dates_prediction_window, flat_labels],
            axis=1,
        )

        # get sessions
        sessions_df = pd.read_csv(Path(__file__).parents[4] / "data/Sessions3.csv")
        sessions_df["startChargeTime"] = pd.to_datetime(sessions_df["startChargeTime"])
        # Round to the nearest 15-minute interval
        sessions_df["startChargeTime"] = sessions_df["startChargeTime"].dt.round(
            "15min"
        )
        # Filter sessions within the prediction window
        sessions_in_samples_range = sessions_df.loc[
            (sessions_df["startChargeTime"] >= start_dates_prediction_window.iloc[0])
            & (
                sessions_df["startChargeTime"] <= start_dates_prediction_window.iloc[-1]
            ),
            ["startChargeTime", "dcosId"],
        ].reset_index(drop=True)

        # Create a dictionary to map dcosId to their corresponding
        # power profiles in the interval data
        number_of_power_columns = power_df.filter(regex=r"power\d+").shape[1]
        session_profiles_from_interval = {}
        for i in range(1, number_of_power_columns + 1):
            dcos_column = f"dcosId{i}"
            power_column = f"power{i}"

            df_power_profiles = (
                power_df[power_df[dcos_column].notna()][
                    [dcos_column, power_column, "date"]
                ]
                .groupby(dcos_column)
                .agg(list)
            )
            df_power_profiles = df_power_profiles.rename(
                columns={power_column: "power_profiles"}
            )
            session_profiles_from_interval.update(
                df_power_profiles.to_dict(orient="index")
            )

        # Initialize an array to store power arrays
        power_arrays = np.full(
            (sessions_in_samples_range.shape[0], max_timesteps_per_session), None
        )

        for index, row in tqdm(
            sessions_in_samples_range.iterrows(),
            "Generating Session Features",
            total=sessions_in_samples_range.shape[0],
        ):
            dcosId = row["dcosId"]

            if dcosId in session_profiles_from_interval:
                session_profiles = session_profiles_from_interval[dcosId]
                power_array = session_profiles["power_profiles"]

                # Here we try to see if the power profile starts at the same time in the
                # interval data and in the sessions data
                start_charge_time_in_intervals = session_profiles["date"][0]
                start_charge_time_in_sessions = row["startChargeTime"]
                time_difference = (
                    start_charge_time_in_intervals - start_charge_time_in_sessions
                )
                # TODO: Do not hardcode the frequency
                if abs(time_difference) >= pd.Timedelta(minutes=15):
                    sessions_in_samples_range.loc[index, "startChargeTime"] = (
                        start_charge_time_in_intervals
                    )

                # Truncate or pad power array to session_max_timesteps elements
                if len(power_array) > max_timesteps_per_session:
                    print(
                        f"WARNING: session {dcosId} power profile truncated to 24 hours because it is {len(power_array)/4} hours long"
                    )
                    power_array = power_array[:max_timesteps_per_session]
                elif len(power_array) < max_timesteps_per_session:
                    power_array = np.pad(
                        power_array,
                        (0, max_timesteps_per_session - len(power_array)),
                        mode="constant",
                    )

                power_arrays[index] = power_array

        # Assign power arrays to sessions_in_samples_range
        for i in range(max_timesteps_per_session):
            sessions_in_samples_range[f"u_{i+1}"] = power_arrays[:, i]

        # Create a dataframe of samples, where we keep only the samples that have
        # a corresponding session
        merged_inputs_dates_sessions = sessions_in_samples_range[
            ["startChargeTime"] + [f"u_{i+1}" for i in range(max_timesteps_per_session)]
        ].merge(
            flat_inputs_dates,
            left_on="startChargeTime",
            right_on="start_date_prediction_window",
            how="inner",
        )

        merged_inputs_dates_sessions = merged_inputs_dates_sessions.dropna()

        # convert the u columns to float, and normalize them by the EVSE max power
        # We need to do that after dropping the NaN values
        # TODO: improve the normalization, this only works for SLRPEV data
        # (if the max power is 6.6kW and there are 8 EVSEs)
        for i in range(max_timesteps_per_session):
            merged_inputs_dates_sessions[f"u_{i+1}"] = merged_inputs_dates_sessions[
                f"u_{i+1}"
            ].astype(float) / (6_600 * 8)

        print(
            f"Number of sessions in DataFrame: {merged_inputs_dates_sessions.shape[0]}"
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
            list(flat_inputs.columns[~flat_inputs.columns.str.contains(r"date_*+")])
            + [f"u_{i+1}" for i in range(max_timesteps_per_session)]
        ]

        if self.peak_prediction:
            flat_labels = self.transform_X_y_for_peak_prediction(
                flat_labels, merged_inputs_dates_sessions
            )

        return flat_inputs, flat_labels, merged_inputs_dates_sessions

    def transform_X_y_for_peak_prediction(
        self,
        flat_labels: pd.DataFrame,
        merged_inputs_dates_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO: Look into predicting only the peak power for the rest of the day
        # (and not the whole day). In this case we only use labels_power_df_from_samples

        # Extract dates from 'startChargeTime'
        dates_to_extract = merged_inputs_dates_sessions["startChargeTime"].dt.date

        # # Group by date and calculate the max 'totalPower' for each date
        # power_df["date_only"] = power_df["date"].dt.date
        # max_power_by_date_old = power_df.groupby("date_only")["power"].max()
        # The functions below compute max_power_by_date from the samples
        # (it gives the same results but at least the data is normalized)

        # Create dataframes of all the samples with 2 columns: 'power' and 'date'
        inputs_power_df_from_samples = pd.DataFrame(
            data=merged_inputs_dates_sessions.filter(regex=r"(power|date)_(\d+)\b")
            .to_numpy()
            .reshape(-1, 2),
            columns=["power", "date"],
        )
        labels_power_df_from_samples = pd.DataFrame(
            data=merged_inputs_dates_sessions.filter(regex=r"(power|date)_(\d+)_label")
            .to_numpy()
            .reshape(-1, 2),
            columns=["power", "date"],
        )
        power_df_from_samples = pd.concat(
            [inputs_power_df_from_samples, labels_power_df_from_samples]
        )
        power_df_from_samples["date"] = convert_date_from_int_to_datetime(
            power_df_from_samples["date"]
        ).dt.date
        # Get the peak power of each day, looking at through the inputs and the labels
        max_power_by_date = power_df_from_samples.groupby("date")["power"].max()

        assert (
            not max_power_by_date.isna().any()
        ), "When trying to extract peak powers, there are NaN values"

        # Extract the max power values for the dates of interest
        y_max = max_power_by_date.reindex(dates_to_extract).values

        # Create the DataFrame with the results
        flat_labels = pd.DataFrame(index=flat_labels.index, data={"y_max": y_max})

        return flat_labels
