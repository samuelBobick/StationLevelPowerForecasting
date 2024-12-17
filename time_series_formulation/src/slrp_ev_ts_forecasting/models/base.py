from typing import Literal, Optional

import numpy as np
import pandas as pd
from slrp_ev_data.window_generator import WindowGenerator
from slrp_ev_ts_forecasting.default_parameters import (
    NUMBER_OF_DAYS_FOR_PACF,
    TypeOptimizeLags,
)
from slrp_ev_ts_forecasting.pacf import get_pacf_values, get_threshold, sort_pacf_values


class Base:
    def __init__(
        self,
        lookahead,
        x_dim,
        optimize_lags: TypeOptimizeLags,
        get_val_data_from_shuffled_train: bool,
    ):
        self.lookahead = lookahead
        self.x_dim = x_dim
        self.optimize_lags = optimize_lags
        self.get_val_data_from_shuffled_train = get_val_data_from_shuffled_train

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
                (y_dates.to_numpy(), forecasts.squeeze(), reals.squeeze()),
                axis=-1,
            )
        else:
            predictions_array = np.stack(
                (y_dates.to_numpy(), forecasts.squeeze()), axis=-1
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
