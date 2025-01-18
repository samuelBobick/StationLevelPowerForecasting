import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots
from slrp_ev_data.feature_engineering import (
    apply_scaling,
    convert_date_from_int_to_datetime,
    get_workday_column_names,
    one_hot_encoding,
)
from slrp_ev_data.read_new_slrpev_data import read_new_slrpev_data
from slrp_ev_data.window_generator import WindowGenerator
from slrp_ev_ts_forecasting.default_parameters import (
    NUMBER_OF_DAYS_FOR_PACF,
    RANDOM_SEED,
    VERBOSE,
    TypeOptimizeLags,
    TypeScalingMode,
)
from slrp_ev_ts_forecasting.helper_session_forecasting import (
    apply_generate_future_session_power,
    extract_features,
    get_artificial_data,
    get_raw_df_sessions,
    revert_power_df,
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
        scaling_mode: TypeScalingMode,
        scaling_parameters: tuple | pd.DataFrame | None,
        session_based_mode: bool,
        peak_prediction: bool,
        add_number_of_sessions: bool,
        add_fraction_of_regular_sessions: bool,
        use_all_active_sessions: bool,
        number_of_artificial_datasets: int,
        random_start_time: bool,
        shuffle_power_profiles: bool,
        random_power_profile_shapes: bool,
        random_user_needs: bool,
        random_choices: bool,
        add_number_of_evses_available: bool,
        verbose: bool = VERBOSE,
    ):
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.optimize_lags = optimize_lags
        self.get_val_data_from_shuffled_train = get_val_data_from_shuffled_train
        self.verbose = verbose
        self.window_seed = (
            RANDOM_SEED if RANDOM_SEED else int(pd.Timestamp.now().timestamp())
        )
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.scaling_mode: TypeScalingMode = scaling_mode
        if scaling_parameters is None:
            raise ValueError(
                f"scaling_parameters should not be None. scaling_mode is {self.scaling_mode}"
            )
        else:
            self.scaling_parameters = scaling_parameters

        self.add_number_of_evses_available = add_number_of_evses_available

        # parameters to get data for session-based forecasting
        self.session_based_mode = session_based_mode
        self.peak_prediction = peak_prediction
        self.add_number_of_sessions = add_number_of_sessions
        self.add_fraction_of_regular_sessions = add_fraction_of_regular_sessions
        self.use_all_active_sessions = use_all_active_sessions

        # parameters to add artificial data
        self.number_of_artificial_datasets = number_of_artificial_datasets
        self.random_start_time = random_start_time
        self.shuffle_power_profiles = shuffle_power_profiles
        self.random_power_profile_shapes = random_power_profile_shapes
        self.random_user_needs = random_user_needs
        self.random_choices = random_choices

        if not session_based_mode and (
            add_number_of_sessions
            or add_fraction_of_regular_sessions
            or use_all_active_sessions
        ):
            warnings.warn(
                "One of the parameters add_number_of_sessions, add_fraction_of_regular_sessions "
                "or use_all_active_sessions is set to True, but those are only used "
                "when session_based_mode=True"
            )

        self.list_workday_column_names = get_workday_column_names(lookahead)

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
        )  # type: ignore
        pacf_df: pd.DataFrame
        interval: float

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
                _val_and_train_window = WindowGenerator(
                    input_width=input_width,
                    label_width=label_width,
                    shift=self.lookahead,
                    train_df=df_padded,
                    get_val_from_shuffled_train=True,
                    label_columns=["power", "date"],
                    overlapping_windows=overlapping_windows,
                    seed=self.window_seed,
                    verbose=True,
                )
                if not hasattr(self, "_val_and_train_window"):
                    self._val_and_train_window = _val_and_train_window
                return _val_and_train_window, _val_and_train_window.train

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
                    "date": convert_date_from_int_to_datetime(
                        pd.Series(predictions_array[i, :, 0])
                    ),
                    "power_0": predictions_array[i, :, 1],
                    "real_power_0": predictions_array[i, :, 2],
                }
            )

            if df_single_prediction["date"].isin(df_predictions["date"]).any():
                # if we already have prediction data for these timesteps, we need to iterate
                # over the other power_x columns to find the first one that doesn't have data yet
                # if they all have data, we create a new column
                df_predictions_these_dates = df_predictions[
                    df_predictions["date"].isin(df_single_prediction["date"])
                ]
                next_power_column_number = (df_predictions.shape[1] - 1) // 2

                # by default we add the data to a new column
                df_single_prediction = df_single_prediction.rename(
                    columns={
                        "power_0": f"power_{next_power_column_number}",
                        "real_power_0": f"real_power_{next_power_column_number}",
                    }
                )
                # If possible, we add it to an existing column
                for j in range(0, next_power_column_number):
                    if df_predictions_these_dates[f"power_{j}"].isna().all():
                        df_single_prediction = df_single_prediction.rename(
                            columns={
                                f"power_{next_power_column_number}": f"power_{j}",
                                f"real_power_{next_power_column_number}": f"real_power_{j}",
                            }
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

        # we have an issue where some dates are duplicated,
        # but with only 1 column that has a value for each duplicate. We actually
        # want only 1 row with a value in each column
        df_predictions = (
            df_predictions.groupby("date")
            .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)
            .reset_index()
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

            # We want the predictions to start at midnight, so we set the first
            # prediction window to start at midnight
            # To do that, we make the data start at midnight - input width/4 hours
            # TODO: change the 4 to be based on the data frequency
            start_hour = 24 - input_width / 4 % 24
            if start_hour == 24:
                start_hour = 0
            dates = convert_date_from_int_to_datetime(df_padded["date"])
            start_date = pd.to_datetime(dates.iloc[0].date()) + pd.Timedelta(
                hours=start_hour
            )
            if start_date not in dates.values:
                start_date += pd.Timedelta(hours=24)
            assert (
                start_date in dates.values
            ), f"start_date {start_date} not in dates {dates}, weird, please debug"

            df_padded = df_padded.loc[dates >= start_date]

        else:
            df_padded = None

        samples = self.make_samples_X_y(
            df_padded,
            time_mode=time_mode,
            data_type=data_type,
            return_y_date=return_y_date,
            overlapping_windows=overlapping_windows,
            multi_model_mode=multi_model_mode,
            input_width=input_width,
            scaling_mode=self.scaling_mode,
            scaling_parameters=self.scaling_parameters,
        )
        if len(samples) == 3:
            return samples
        else:
            flat_inputs, flat_labels = samples

        if data_type == "train":
            if return_y_date:
                raise ValueError(
                    "return_y_date is not yet supported for 'train' data. Please set it to False"
                )
            if df_padded is None:
                raise ValueError(
                    "df_padded should be provided to generate windows for train data type"
                )

            for i in range(self.number_of_artificial_datasets):
                (
                    artificial_train_data,
                    artificial_raw_df_sessions,
                    scaling_parameters,
                ) = get_artificial_data(
                    train_data=df_padded,
                    random_start_time=self.random_start_time,
                    shuffle_power_profiles=self.shuffle_power_profiles,
                    random_power_profile_shapes=self.random_power_profile_shapes,
                    random_user_needs=self.random_user_needs,
                    random_choices=self.random_choices,
                    scaling_mode=self.scaling_mode,
                    lookahead=self.lookahead,
                    rng=self.rng,
                )

                artificial_flat_inputs, artificial_flat_labels = self.make_samples_X_y(  # type: ignore
                    artificial_train_data,
                    time_mode=time_mode,
                    data_type=data_type,
                    return_y_date=return_y_date,
                    overlapping_windows=overlapping_windows,
                    multi_model_mode=multi_model_mode,
                    input_width=input_width,
                    artificial_raw_df_sessions=artificial_raw_df_sessions,
                    scaling_mode=self.scaling_mode,
                    scaling_parameters=scaling_parameters,
                )
                flat_inputs = pd.concat([flat_inputs, artificial_flat_inputs])
                flat_labels = pd.concat([flat_labels, artificial_flat_labels])

            # Shuffle the data
            indices = flat_inputs.index.to_numpy(copy=True)
            self.rng.shuffle(indices)
            flat_inputs = flat_inputs.loc[indices]
            flat_labels = flat_labels.loc[indices]

        return flat_inputs, flat_labels

    def make_samples_X_y(
        self,
        df_padded: pd.DataFrame | None,
        time_mode: Literal["window", "cyclical"],
        data_type: Literal["train", "val", "test"],
        return_y_date: bool,
        overlapping_windows: bool,
        multi_model_mode: bool,
        input_width: int,
        scaling_mode: TypeScalingMode,
        scaling_parameters: tuple | pd.DataFrame,
        artificial_raw_df_sessions: pd.DataFrame | None = None,
    ):
        W, window_data = self.get_window_data(
            df_padded, input_width, self.lookahead, overlapping_windows, data_type
        )

        cols_keep_last_value = self.list_workday_column_names.copy()
        if self.add_number_of_evses_available:
            cols_keep_last_value += ["number_of_evses_available"]
        if time_mode == "cyclical":
            cols_keep_last_value += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ]
        elif time_mode == "window":
            cols_keep_last_value += ["time_window"]

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
                    artificial_raw_df_sessions=artificial_raw_df_sessions,
                    scaling_mode=scaling_mode,
                    scaling_parameters=scaling_parameters,
                )
            )
            if self.verbose:
                self.plot_example_sample(
                    merged_inputs_dates_sessions,
                    flat_inputs,
                    flat_labels,
                    data_type=data_type,
                    date="2024-02-26" if data_type == "test" else None,
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
        artificial_raw_df_sessions: pd.DataFrame | None,
        scaling_mode: TypeScalingMode,
        scaling_parameters: tuple | pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: Change the loss function in the model, to better predict peaks

        start_dates_prediction_window = (
            pd.to_datetime(flat_labels["date_0"], unit="s")
            .dt.round("15min")
            .rename("start_date_prediction_window")
        )

        if artificial_raw_df_sessions is None:
            power_df = read_new_slrpev_data(keep_all_columns=True)
            power_df = power_df.loc[
                (power_df["date"].dt.date >= start_dates_prediction_window.min().date())
                & (
                    power_df["date"].dt.date
                    <= start_dates_prediction_window.max().date()
                )
            ]
        else:
            power_df = None

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

        df_sessions = self.get_df_sessions(
            power_df,
            artificial_raw_df_sessions,
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
        )

        # Create a dataframe of samples, where we keep only the samples that have
        # a corresponding session
        merged_inputs_dates_sessions = df_sessions.merge(
            flat_inputs_dates,
            left_on="startChargeTime",
            right_on="start_date_prediction_window",
            how="inner",
        )

        merged_inputs_dates_sessions = merged_inputs_dates_sessions.dropna()

        additional_feature_names = []
        # Add number of session and fraction or regular in the features
        if self.add_number_of_sessions:
            additional_feature_names += ["numberOfActiveSessions"]
        if self.add_fraction_of_regular_sessions:
            additional_feature_names += ["fractionOfRegularSessions"]

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
            + additional_feature_names
        ]

        if self.peak_prediction:
            flat_labels = self.transform_X_y_for_peak_prediction(
                merged_inputs_dates_sessions
            )

        return flat_inputs, flat_labels, merged_inputs_dates_sessions

    def get_df_sessions(
        self,
        power_df: pd.DataFrame | None,
        artificial_raw_df_sessions: pd.DataFrame | None,
        scaling_mode: TypeScalingMode,
        scaling_parameters: tuple | pd.DataFrame,
    ) -> pd.DataFrame:
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
        if artificial_raw_df_sessions is not None:
            raw_df_sessions = artificial_raw_df_sessions
        else:
            if power_df is None:
                raise ValueError(
                    "power_df should be provided to generate the sessions DataFrame"
                )
            raw_df_sessions = get_raw_df_sessions(power_df)

        raw_df_sessions["startChargeTime"] = raw_df_sessions.apply(
            lambda x: x["date"][0], axis=1
        )
        raw_df_sessions["endChargeTime"] = raw_df_sessions.apply(
            lambda x: x["date"][-1], axis=1
        )

        reverted_power_df = revert_power_df(raw_df_sessions)
        additional_session_features = extract_features(
            reverted_power_df=reverted_power_df
        ).drop(columns=["power"])
        if not self.add_number_of_sessions:
            additional_session_features = additional_session_features.drop(
                columns=["numberOfActiveSessions"]
            )
        if not self.add_fraction_of_regular_sessions:
            additional_session_features = additional_session_features.drop(
                columns=["fractionOfRegularSessions"]
            )

        print("Generating future session power profiles")
        df_sessions = raw_df_sessions.progress_apply(
            apply_generate_future_session_power,
            axis=1,
            raw_df_sessions=raw_df_sessions,
            use_all_active_sessions=self.use_all_active_sessions,
            lookahead=self.lookahead,
        )  # type: ignore
        # We have a small issues with missing vales in the power profile when
        # the charge ends before the user unplugs the car. We fill the missing values
        # with 0
        df_sessions = df_sessions.fillna(0)

        # scale u columns
        # We need to do that after dropping the NaN values
        list_u_columns = df_sessions.filter(regex=r"u_").columns
        for u_col in list_u_columns:
            df_sessions_for_scaling = df_sessions[[u_col, "startChargeTime"]].rename(
                {u_col: "power", "startChargeTime": "date"}, axis=1
            )
            df_sessions[u_col] = apply_scaling(
                df_sessions_for_scaling, scaling_mode, scaling_parameters, ["power"]
            )["power"]

        df_sessions = pd.merge(
            df_sessions,
            additional_session_features,
            on="startChargeTime",
            how="left",
        )

        return df_sessions

    def transform_X_y_for_peak_prediction(
        self,
        merged_inputs_dates_sessions: pd.DataFrame,
        mode: Literal["peak_of_day", "peak_next_8h"] = "peak_next_8h",
    ) -> pd.DataFrame:
        if mode == "peak_of_day":
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
                    "power": merged_inputs_dates_sessions.filter(
                        regex=r"(power)_(\d+)\b"
                    )
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
                    "date": merged_inputs_dates_sessions.filter(
                        regex=r"(date)_(\d+)_label"
                    )
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
            flat_labels = pd.DataFrame(
                index=merged_inputs_dates_sessions.index, data={"peak_power": y_max}
            )

        elif mode == "peak_next_8h":
            # Get the peak power of the next 12 hours
            label_columns = [
                col
                for col in merged_inputs_dates_sessions.filter(
                    regex=r"(power)_(\d+)_label"
                ).columns
                if int(col.split("_")[1]) < 8 * 4
            ]
            flat_labels = (
                merged_inputs_dates_sessions[label_columns].max(axis=1).to_frame()
            )

        else:
            raise ValueError("mode should be 'peak_of_day' or 'peak_next_12h'")

        return flat_labels

    def plot_example_sample(
        self,
        merged_inputs_dates_sessions,
        flat_inputs,
        flat_labels,
        data_type: str,
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
            date = merged_inputs_dates_sessions["startChargeTime"].dt.date.iloc[
                -7 * 24 * 4
            ]
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
            # column_names_of_non_lagged_features = flat_inputs.filter(regex=r"^(?!power_\d+$|u_\d+$)")
            subtitle = ""
            if "fractionOfRegularSessions" in flat_inputs.columns:
                subtitle += f"Fraction of regular sessions: {flat_inputs['fractionOfRegularSessions'].iloc[i]:.2f}\n"
            if "numberOfActiveSessions" in flat_inputs.columns:
                subtitle += f"Number of active sessions: {flat_inputs['numberOfActiveSessions'].iloc[i] * 8}\n"
            if "number_of_evses_available" in flat_inputs.columns:
                subtitle += f"Scaled number of EVSEs available: {flat_inputs['number_of_evses_available'].iloc[i]}\n"
            for workday_column in self.list_workday_column_names:
                if workday_column in flat_inputs.columns:
                    subtitle += f"Workday: {flat_inputs[workday_column].iloc[i]}\n"

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
            title_text=f"Example sample of the {data_type} data for date {date}",
            showlegend=True,
            yaxis_title="Normalized Power",
        )
        fig.show()
