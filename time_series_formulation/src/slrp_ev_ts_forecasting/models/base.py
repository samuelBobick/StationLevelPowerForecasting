from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots
from slrp_ev_data.feature_engineering import (
    convert_date_from_int_to_datetime,
    one_hot_encoding,
)
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
)
from slrp_ev_data.read_new_slrpev_data import read_new_slrpev_data
from slrp_ev_data.window_generator import WindowGenerator
from slrp_ev_ts_forecasting.default_parameters import (
    NUMBER_OF_DAYS_FOR_PACF,
    TypeOptimizeLags,
)
from slrp_ev_ts_forecasting.pacf import get_pacf_values, get_threshold, sort_pacf_values
from tqdm.auto import tqdm

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas()


class Base:
    def __init__(
        self,
        x_dim,
        lookahead,
        optimize_lags: TypeOptimizeLags,
        get_val_data_from_shuffled_train: bool,
        session_based_mode: bool,
        peak_prediction: bool,
        add_number_of_sessions: bool,
        add_fraction_of_regular_sessions: bool,
        use_all_active_sessions: bool,
    ):
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.optimize_lags = optimize_lags
        self.get_val_data_from_shuffled_train = get_val_data_from_shuffled_train

        # parameters to get data for session-based forecasting
        self.session_based_mode = session_based_mode
        self.peak_prediction = peak_prediction
        self.add_number_of_sessions = add_number_of_sessions
        self.add_fraction_of_regular_sessions = add_fraction_of_regular_sessions
        self.use_all_active_sessions = use_all_active_sessions

        # initialize parameters defined in the child classes
        self.pacf_top_values = None

    def get_top_pacf_values(
        self, data: pd.DataFrame, nb_of_days_for_pacf: int = NUMBER_OF_DAYS_FOR_PACF
    ) -> pd.Series:
        """Returns a list-like object with the top x_dim values of the Partial AutoCorrelation Function (PACF)."""
        # Define "number of steps to predict" given to the PACF function
        if self.optimize_lags == "long_opt":
            nb_of_steps_to_predict = 1
        elif self.optimize_lags == "short_opt":
            nb_of_steps_to_predict = self.lookahead
        else:
            raise ValueError("Optimize_lags should be 'short_opt' or 'long_opt'")

        pacf_df, interval = get_pacf_values(
            downsample_hours=1,
            data=data,
            nb_of_days_for_pacf=nb_of_days_for_pacf,
            nb_of_steps_to_predict=nb_of_steps_to_predict,
            return_confidence_interval=True,
        )

        number_of_lags_to_keep = self.x_dim
        pacf_top_values = sort_pacf_values(pacf_df, number_of_lags_to_keep)

        _, self.index_farthest_lag = get_threshold(
            pacf_df, number_of_lags_to_keep=number_of_lags_to_keep, interval=interval
        )
        print(f"Farthest lag: {self.index_farthest_lag / 96} days back")
        return pacf_top_values["PACF"]

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

    def get_window_data(
        self,
        df_padded: Optional[pd.DataFrame],
        input_width: int,
        label_width: int,
        overlapping_windows: bool,
        data_type: Literal["train", "val", "test"],
    ) -> tuple[WindowGenerator, list[tuple]]:

        if self.get_val_data_from_shuffled_train:
            if data_type == "train":
                # in this case, we save the window with the full data to be able to
                # generate the validation set later
                if df_padded is None:
                    raise ValueError(
                        "df_padded should be provided to generate windows "
                        "for train and test data types"
                    )
                self._val_and_train_window = WindowGenerator(
                    input_width=input_width,
                    label_width=label_width,
                    shift=self.lookahead,
                    train_df=df_padded,
                    get_val_from_shuffled_train=True,
                    label_columns=["power", "date"],
                    overlapping_windows=overlapping_windows,
                    verbose=True,
                )
                return self._val_and_train_window, self._val_and_train_window.train
            elif data_type == "val":
                if df_padded is not None:
                    raise ValueError(
                        "df_padded should be None when getting the validation data "
                        "from the shuffled train data"
                    )
                return self._val_and_train_window, self._val_and_train_window.val

        if df_padded is None:
            raise ValueError(
                "df_padded should be provided to generate windows "
                "for train and test data types"
            )
        # default case
        W = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=self.lookahead,
            train_df=df_padded,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
        )
        window_data = W.train

        return W, window_data

    def prepare_df_predictions(
        self, forecasts: np.ndarray, y_dates, reals: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        # TODO: use it in all the other predict functions
        add_real = reals is not None
        if add_real:
            predictions_array = np.stack(
                (y_dates.to_numpy().squeeze(), forecasts.squeeze(), reals.squeeze()),
                axis=-1,
            )
        else:
            predictions_array = np.stack(
                (y_dates.to_numpy().squeeze(), forecasts.squeeze()), axis=-1
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
            if add_real:
                df_single_prediction["real_power"] = predictions_array[i, :, 2]

            if df_single_prediction["date"].isin(df_predictions["date"]).any():
                # if we already have prediction data for these timesteps, we need to iterate
                # over the other power_x columns to find the first one that doesn't have data yet
                # if they all have data, we create a new column
                df_predictions_these_dates = df_predictions[
                    df_predictions["date"].isin(df_single_prediction["date"])
                ]
                next_power_column_number = len(df_predictions.columns) - 1
                if add_real:
                    next_power_column_number -= 1
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
        return df_predictions

    def save_model(self, model, model_name: str):
        print(
            "WARNING: Model not saved. save_model method not implemented for this model."
        )

    @property
    def model_str_name(self) -> str:
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def get_X_y(
        self,
        df: pd.DataFrame | None,
        time_mode: Literal["window", "cyclical"],
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
        multi_model_mode: bool = True,
    ):
        print(f"## Getting {data_type} data")
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

        cols_keep_last_value = []
        if time_mode == "cyclical":
            cols_keep_last_value += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ]
            if multi_model_mode:
                cols_keep_last_value += ["workday"]
        elif time_mode == "window":
            cols_keep_last_value += ["time_window", "workday"]

        label_cols_to_flatten = ["power"]
        if self.session_based_mode:
            label_cols_to_flatten += ["date"]

        input_cols_to_flatten = ["power"]
        if self.session_based_mode:
            input_cols_to_flatten += ["date"]

        if self.optimize_lags:
            if self.pacf_top_values is None:
                raise ValueError(
                    "pacf_top_values is undefined, it needs to be defined "
                    "in child classes, check your code"
                )
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
        mask_nan = flat_inputs.isna().any(axis=1) | flat_labels.isna().any(axis=1)
        flat_inputs = flat_inputs[~mask_nan]
        flat_labels = flat_labels[~mask_nan]

        if multi_model_mode and time_mode == "window":
            flat_inputs = one_hot_encoding(flat_inputs, ["time_window"])

        if self.session_based_mode:
            flat_inputs, flat_labels, merged_inputs_dates_sessions = (
                self.transform_X_y_for_session_based_mode(
                    flat_inputs,
                    flat_labels,
                )
            )
            if data_type == "test":
                self.plot_example_sample(
                    merged_inputs_dates_sessions,
                    flat_inputs,
                    flat_labels,
                    date="2024-02-26",
                )

        print(
            f"Data has {flat_inputs.shape[0]} samples, with {flat_inputs.shape[1]} features, to predict {flat_labels.shape[1]} labels"
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
                y_dates = y_dates[~mask_nan]
            return flat_inputs, flat_labels, y_dates
        else:
            return flat_inputs, flat_labels

    def transform_X_y_for_session_based_mode(
        self,
        flat_inputs: pd.DataFrame,
        flat_labels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: Also add the number of current active sessions as a feature? Information is
        # maybe already in the "load until the start of this session"
        # TODO: Add the fraction of scheduled users in the current sessions?
        # TODO: Change the loss function in the model, to better predict peaks

        start_dates_prediction_window = (
            pd.to_datetime(flat_labels["date_0"], unit="s")
            .dt.round("15min")
            .rename("start_date_prediction_window")
        )

        power_df = read_new_slrpev_data(keep_all_columns=True)
        power_df = power_df.loc[power_df["date"].isin(start_dates_prediction_window)]

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

        df_sessions = self.get_df_sessions(power_df)

        # Create a dataframe of samples, where we keep only the samples that have
        # a corresponding session
        merged_inputs_dates_sessions = df_sessions.merge(
            flat_inputs_dates,
            left_on="startChargeTime",
            right_on="start_date_prediction_window",
            how="inner",
        )

        merged_inputs_dates_sessions = merged_inputs_dates_sessions.dropna()

        number_of_sessions_feature_names = []
        if self.add_number_of_sessions or self.add_fraction_of_regular_sessions:
            # Add number of session and fraction or regular in the features
            # TODO: if we generate random session, see how to modify the fraction
            # of regular sessions. Maybe we don't generate random session in place of
            # sessions that were regular
            if self.add_number_of_sessions:
                number_of_sessions_feature_names += ["numberOfActiveSessions"]
            if self.add_fraction_of_regular_sessions:
                number_of_sessions_feature_names += ["fractionOfRegularSessions"]

            merged_inputs_dates_sessions = merged_inputs_dates_sessions.merge(
                power_df[number_of_sessions_feature_names + ["date"]],
                left_on="startChargeTime",
                right_on="date",
                how="left",
            )
            merged_inputs_dates_sessions = merged_inputs_dates_sessions.drop(
                columns="date"
            )

        merged_inputs_dates_sessions = merged_inputs_dates_sessions.sort_values(
            by="startChargeTime"
        )

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
            list(flat_inputs.columns[~flat_inputs.columns.str.contains(r"date")])
            + [f"u_{i+1}" for i in range(self.lookahead)]
            + number_of_sessions_feature_names
        ]

        if self.peak_prediction:
            flat_labels = self.transform_X_y_for_peak_prediction(
                flat_labels, merged_inputs_dates_sessions
            )

        return flat_inputs, flat_labels, merged_inputs_dates_sessions

    def get_df_sessions(self, power_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a DataFrame of sessions from a power DataFrame.
        The data frame has the dcosId (session ID) as index, and the future power \
        profile of the session as `self.lookahead` columns. It also has \
        the start and end charge times of the session.

        Args:
            power_df (pd.DataFrame): dataframe read from the function `read_new_slrpev_data`.
        Returns:
            pd.DataFrame: The DataFrame of sessions with `self.lookahead` + 2.
        """
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

        raw_df_sessions = pd.DataFrame(session_profiles_from_interval).T
        raw_df_sessions["startChargeTime"] = raw_df_sessions.apply(
            lambda x: x["date"][0], axis=1
        )
        raw_df_sessions["endChargeTime"] = raw_df_sessions.apply(
            lambda x: x["date"][-1], axis=1
        )
        # keep meaningful sessions
        # filter sessions to keep only the ones with a total energy charged greater than 2kWh
        # We convert to Series to have a sum even if the list contains NaN values
        # We divide by 4 the sum of interval power values to get an energy value
        # TODO: do not hardcode the 4 (use frequency of the data instead)
        raw_df_sessions = raw_df_sessions.loc[
            raw_df_sessions.apply(
                lambda x: pd.Series(x["power_profiles"]).sum(), axis=1
            )
            / 4
            > 2000
        ]

        def _apply_generate_future_session_power(row):
            dcosId = row.name
            current_time = row["startChargeTime"]

            if self.use_all_active_sessions:
                data = _sum_all_active_sessions(raw_df_sessions, current_time)
            else:
                data = row["power_profiles"]

            # Truncate or pad power array to self.lookahead elements
            if len(data) > self.lookahead:
                data = data[: self.lookahead]
            elif len(data) < self.lookahead:
                data = np.pad(
                    data,
                    (0, self.lookahead - len(data)),
                    mode="constant",
                )

            new_row = pd.Series(
                data=data,
                index=[f"u_{i+1}" for i in range(self.lookahead)],
                name=dcosId,
                dtype=float,
            )
            new_row["startChargeTime"] = current_time
            return new_row

        df_sessions = raw_df_sessions.progress_apply(
            _apply_generate_future_session_power, axis=1
        )  # type: ignore
        # We have a small issues with missing vales in the power profile when
        # the charge ends before the user unplugs the car. We fill the missing values
        # with 0
        df_sessions = df_sessions.fillna(0)

        # normalize u columns by the EVSE max power
        # We need to do that after dropping the NaN values
        # TODO: improve the normalization, this only works for SLRPEV data
        # (if the max power is 6.6kW and there are 8 EVSEs)
        if self.use_all_active_sessions:
            for i in range(self.lookahead):
                df_sessions[f"u_{i+1}"] /= SINGLE_EVSE_NORMALIZATION_PARAM * 8
        else:
            for i in range(self.lookahead):
                df_sessions[f"u_{i+1}"] /= SINGLE_EVSE_NORMALIZATION_PARAM

        return df_sessions

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
            data={
                "power": merged_inputs_dates_sessions.filter(regex=r"(power)_(\d+)\b")
                .to_numpy()
                .flatten(),
                "date": merged_inputs_dates_sessions.filter(regex=r"(date)_(\d+)\b")
                .to_numpy()
                .flatten(),
            }
        )
        labels_power_df_from_samples = pd.DataFrame(
            data={
                "power": merged_inputs_dates_sessions.filter(
                    regex=r"(power)_(\d+)_label"
                )
                .to_numpy()
                .flatten(),
                "date": merged_inputs_dates_sessions.filter(regex=r"(date)_(\d+)_label")
                .to_numpy()
                .flatten(),
            }
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
        flat_labels = pd.DataFrame(index=flat_labels.index, data={"peak_power": y_max})

        return flat_labels

    def plot_example_sample(
        self,
        merged_inputs_dates_sessions,
        flat_inputs,
        flat_labels,
        date=None,
        number_of_samples: int = 3,
    ):
        """Plot a plotly example sample of the data."""
        fig = subplots.make_subplots(
            rows=number_of_samples,
            cols=1,
            shared_xaxes=True,
        )

        if date is None:
            date = merged_inputs_dates_sessions["startChargeTime"].dt.date.iloc[0]
        else:
            date = pd.to_datetime(date).date()
            if (
                date
                not in merged_inputs_dates_sessions["startChargeTime"].dt.date.unique()
            ):
                raise ValueError(
                    f"Date {date} not found in the data. Available dates are from "
                    f"{merged_inputs_dates_sessions['startChargeTime'].dt.date.min()}"
                    f" to {merged_inputs_dates_sessions['startChargeTime'].dt.date.max()}"
                )

        # Filter the data for the given date
        assert (merged_inputs_dates_sessions.index == flat_inputs.index).all()
        assert (merged_inputs_dates_sessions.index == flat_labels.index).all()
        merged_inputs_dates_sessions = merged_inputs_dates_sessions[
            merged_inputs_dates_sessions["startChargeTime"].dt.date >= date
        ]
        flat_inputs = flat_inputs.loc[merged_inputs_dates_sessions.index]
        flat_labels = flat_labels.loc[merged_inputs_dates_sessions.index]

        for i in range(number_of_samples):
            show_legend = i == 0

            fig.add_trace(
                go.Scatter(
                    x=convert_date_from_int_to_datetime(
                        merged_inputs_dates_sessions.filter(regex=r"date_\d+\b").iloc[i]
                    ),
                    y=flat_inputs.filter(regex=r"power_\d+").iloc[i],
                    mode="lines",
                    name="Feat - Historical aggregated power",
                    line=dict(color="lightblue"),
                    showlegend=show_legend,
                ),
                row=i + 1,
                col=1,
            )

            u_date_range = pd.date_range(
                start=merged_inputs_dates_sessions["startChargeTime"].iloc[i],
                periods=self.lookahead,
                freq="15min",
            )
            scale_factor_future_sessions = 1
            if not self.use_all_active_sessions:
                scale_factor_future_sessions = 8
            fig.add_trace(
                go.Scatter(
                    x=u_date_range,
                    y=flat_inputs.filter(regex=r"u_\d+").iloc[i]
                    / scale_factor_future_sessions,
                    mode="lines",
                    name="Feat - Session power",
                    line=dict(color="red", dash="dash"),
                    showlegend=show_legend,
                ),
                row=i + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=convert_date_from_int_to_datetime(
                        merged_inputs_dates_sessions.filter(
                            regex=r"date_\d+_label\b"
                        ).iloc[i]
                    ),
                    y=flat_labels.iloc[i],
                    mode="markers",
                    name="Label - Future aggregated power",
                    line=dict(color="yellow"),
                    showlegend=show_legend,
                ),
                row=i + 1,
                col=1,
            )

            # Add subtitle for each subplot
            subtitle = ""
            if "fractionOfRegularSessions" in flat_inputs.columns:
                subtitle += f"Fraction of regular sessions: {flat_inputs['fractionOfRegularSessions'].iloc[i]:.2f}\n"
            if "numberOfActiveSessions" in flat_inputs.columns:
                subtitle += f"Number of active sessions: {flat_inputs['numberOfActiveSessions'].iloc[i]}\n"

            fig.add_annotation(
                text=subtitle,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1 - (i / number_of_samples),
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12),
            )

        fig.update_layout(
            title_text=f"Example sample of the data for date {date}",
            showlegend=True,
            yaxis_title="Normalized Power",
        )
        fig.show()


def _sum_all_active_sessions(
    raw_df_sessions: pd.DataFrame, current_time: pd.Timestamp
) -> list:
    # We are going to replace the u features (future profile of the next session)
    # with the aggregated future power profiles of all the active sessions
    active_sessions = raw_df_sessions[
        (raw_df_sessions["startChargeTime"] <= current_time)
        & (raw_df_sessions["endChargeTime"] >= current_time)
    ]
    sum_df = pd.DataFrame()
    for _, this_session in active_sessions.iterrows():
        this_session_df = pd.DataFrame(
            data=this_session["power_profiles"], index=this_session["date"]
        )
        sum_df = pd.concat([sum_df, this_session_df], axis=1)
    sum_df = sum_df.sum(axis=1)
    sum_df = sum_df.loc[sum_df.index >= current_time]
    return sum_df.to_list()
