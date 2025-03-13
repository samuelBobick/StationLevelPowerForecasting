import os
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from plotly import graph_objects as go
from torch.utils.data import Dataset

from slrp_ev_data.utils.data_utils import (
    convert_data_freq_to_minutes,
    convert_date_from_int_to_datetime,
    get_data_frequency,
)

# source: https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

# Set the logging level using TensorFlow's logging module, to only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Window Generator is using device: {DEVICE}")


class TypeKeepSomeValues(TypedDict):
    col_name: str
    indexes_to_keep: list[int] | np.ndarray


class TFToTorchDataset(Dataset):
    def __init__(
        self,
        tf_dataset: list[tuple],
        keep_last_value_indices: list[int],
        keep_as_label_indices: list[int],
    ):
        self.inputs = []
        self.labels = []

        # Iterate over TensorFlow batches and flatten to individual samples
        for batch in tf_dataset:
            inputs, labels = batch
            inputs = inputs.numpy()  # Convert TensorFlow tensors to NumPy arrays
            labels = labels.numpy()  # Convert TensorFlow tensors to NumPy arrays
            for i in range(len(inputs)):  # Unpack individual samples from the batch
                self.inputs.append(inputs[i][:, keep_last_value_indices])
                self.labels.append(labels[i][:, keep_as_label_indices])

        # Convert lists to NumPy arrays after the loop to avoid pytorch warning
        # about slow tensor creation from list of array
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        This is required for PyTorch's DataLoader.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index. This is required for PyTorch's DataLoader.
        Returns:
        - input_tensor: Input tensor for the specific sample
        - label_tensor: Label tensor for the specific sample
        """
        input_tensor = torch.tensor(
            self.inputs[idx], dtype=torch.float32, device=DEVICE
        )
        label_tensor = torch.tensor(
            self.labels[idx], dtype=torch.float32, device=DEVICE
        )
        return input_tensor, label_tensor

    def get_full_data(self):
        """
        Returns:
        - input_tensor: Full dataset input as a PyTorch tensor
        - label_tensor: Full dataset labels as a PyTorch tensor
        """
        # Convert lists of arrays to PyTorch tensors
        input_tensor = torch.tensor(self.inputs, dtype=torch.float32, device=DEVICE)
        label_tensor = torch.tensor(self.labels, dtype=torch.float32, device=DEVICE)
        return input_tensor, label_tensor


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        get_val_from_shuffled_train: bool = False,
        label_columns: list[str] | None = None,
        overlapping_windows: bool = False,
        batch_size: int = 64,
        seed: int | None = None,
        verbose=True,
    ):
        """_summary_

        Args:
            input_width (int): Number of time steps to include in the input window.
            label_width (int): Number of time steps to include in the label window.
            shift (int): Number of time steps to shift the start of the label window \
                relative to the start of the input window. Usually equal to input_width \
                if you don't want to overlap input and label windows.
            train_df (pd.DataFrame): The training dataset, in the \
                "FeaturedEngineeredSchema" format.
            test_df (pd.DataFrame | None, optional): The test dataset, \
                in the "FeaturedEngineeredSchema" format. Defaults to None.
            val_df (pd.DataFrame | None, optional): The validation dataset, in the \
                "FeaturedEngineeredSchema" format. Defaults to None.
            get_val_from_shuffled_train (bool, optional): Whether to get the validation \
                set from the shuffled training set. If True, don't generate a validation \
                set with your data (do a split between train and test only), and set \
                val_df to None. Defaults to False. 
            label_columns (list[str] | None, optional): List of column names to use \
                as labels. Defaults to None.
            overlapping_windows (bool, optional): Whether to have overlapping label \
                windows in the data. If the shift is greater than the label width, \
                inputs will overlap (which is fine!). Defaults to False.
            batch_size (int, optional): Defaults to 32. For now, set this to 1 if \
                you want to use flatten_dataset.
            verbose (bool, optional): Whether to print debug information. \
                Defaults to True.
        
        Example:
        >>> w1 = WindowGenerator(
                input_width=x_dim,
                label_width=x_dim,
                shift=lookahead,
                train_df=feature_engineering(df_train, normalize_parameters=normalize_params),
                test_df=feature_engineering(df_test, normalize_parameters=normalize_params),
                get_val_from_shuffled_train=True,
                label_columns=["power", "date"],
                overlapping_windows=False,
                batch_size=1000,
                verbose=True,
            )
        """
        self.verbose = verbose

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        if get_val_from_shuffled_train and (val_df is not None):
            raise ValueError(
                "You cannot specify val_df if you want to get it from train_df. "
                "When using get_val_from_shuffled_train = True please set val_df "
                "to None. You must not generate a validation set before."
            )
        self.get_val_from_shuffled_train = get_val_from_shuffled_train
        self.test_df = test_df

        self.data_freq = get_data_frequency(train_df)
        self.data_freq_minutes = convert_data_freq_to_minutes(self.data_freq)

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.sequence_stride = self.get_sequence_stride(overlapping_windows)
        self.batch_size = batch_size

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.seed = seed

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
                f"\nFor the second window, inputs will start at {self.input_indices[0]+self.sequence_stride} "
                + f"and end at {self.input_indices[-1]+self.sequence_stride},",
                "Use the .plot function to see the data.",
            ]
        )

    def split_window(self, features):
        # shape of features is (batch_size, total_window_size, num_features)
        # Here, we filter out the windows that have a gap in the data
        dates = features[:, :, self.column_indices["date"]]
        date_diffs = dates[:, 1:] - dates[:, :-1]
        # A gap is defined as a difference of more than 2 times the
        # data frequency (indeed, if there is no gap,
        # the difference should be the data frequency)
        # Note that due to rounding, the difference is not
        # always strictly equal to the data frequency
        # for instance, if the data frequency is 15 minutes, some differences
        # can be 14 minutes or others 17 minutes
        threshold_for_gap = self.data_freq_minutes * 60 * 2 * 0.9
        # The reduce all operation is to check if all the differences in a
        # window are less than the threshold. If not, the results for this window
        # is "False"
        continuous_mask = tf.reduce_all(
            tf.math.less(date_diffs, threshold_for_gap), axis=1
        )
        # We count the number of gaps in the window
        continuous_features = tf.boolean_mask(features, continuous_mask)

        inputs = continuous_features[:, self.input_slice, :]
        labels = continuous_features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffle: bool = False) -> list[tuple]:
        data = np.array(data, dtype=np.float32)
        max_number_of_windows = np.floor(
            (data.shape[0] - self.total_window_size + self.sequence_stride)
            / self.sequence_stride
        )
        if self.verbose:
            print(
                f"Data length: {data.shape[0]:.0f}. "
                f"We should have {max_number_of_windows:.0f} windows"
            )

        ds = tf.keras.utils.timeseries_dataset_from_array(  # type: ignore
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=shuffle,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        ds = ds.map(self.split_window)
        number_of_windows = self.get_dataset_size(ds)
        if self.verbose:
            gaps = (
                max_number_of_windows - number_of_windows
            )  # counter for the number of gaps in the data
            if gaps > 0:
                percentage_gaps = gaps / max_number_of_windows
                print(
                    f"WARNING: Number of gaps (=number of windows dropped) found "
                    f"when making windows: {gaps:.0f} ({percentage_gaps:.2%}% of the data)"
                )
                if percentage_gaps > 0.6:
                    raise ValueError(
                        "The number of gaps is too high. "
                        "Please reduce the amount of missing the values "
                        "or the size of the windows."
                    )

        # return dataset as eager tensor
        return list(ds)

    def get_dataset_size(self, data: list[tuple]) -> int:
        # Count number of samples
        number_of_samples = 0
        for input, label in data:
            number_of_samples += input.shape[0]
        return number_of_samples

    def split_dataset(
        self, data: list[tuple], fraction_in_val: float = 0.22
    ) -> tuple[list[tuple], list[tuple]]:
        """Split the dataset into a training and validation set.
        
        Args:
            fraction_in_val (float, optional): Fraction of the dataset to \
                use for validation. Default to 0.22, which corresponds to the same \
                (if the train, test split is at 0.9, 0.1) \
                as when we do a train, val, test split of 0.7, 0.2, 0.1.
        """
        number_of_samples = self.get_dataset_size(data)
        number_of_batches_in_val = np.ceil(
            number_of_samples * fraction_in_val / self.batch_size
        ).astype(int)
        # we want at least 1 batch in the validation set
        number_of_batches_in_val = max(1, number_of_batches_in_val)
        if self.verbose:
            print(
                f"Splitting Dateset: "
                f"We want around {number_of_samples*fraction_in_val:.0f} samples in the validation set.\n"
                f"We have a batch size of {self.batch_size}. "
                f"Thus, we will have {number_of_batches_in_val} batches in the validation set."
            )

        val_data = data[-number_of_batches_in_val:]
        train_data = data[:-number_of_batches_in_val]

        # check for number of duplicates
        df_val = pd.DataFrame()
        for _, labels in val_data:
            df_val = pd.concat([df_val, pd.DataFrame(labels.numpy()[:, :, 1])])

        df_train = pd.DataFrame()
        for _, labels in train_data:
            df_train = pd.concat([df_train, pd.DataFrame(labels.numpy()[:, :, 1])])

        df_all = pd.concat([df_val, df_train])

        assert (
            df_all.duplicated().sum() == 0
        ), f"There are duplicates {df_all.duplicated().sum()} between train and val."
        # End of check

        if self.verbose:
            print(
                f"There are actually {self.get_dataset_size(val_data)} samples in "
                f"the validation set and {self.get_dataset_size(train_data)} "
                "samples in the training set."
            )

        return train_data, val_data

    def flatten_dataset(
        self,
        data: list[tuple],
        cols_to_flatten: list[str] = [],
        cols_keep_last_value: list[str] = [],
        cols_keep_some_values: list[TypeKeepSomeValues] = [],
        label_cols_to_flatten: list[str] = [],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """For each example, flattens the 2D input data into a 1D array.
        This version is more efficient than the previous one and works with batch size > 1.
        Use larger batch size to speed up the computation.

        Args:
            data (output of WindowGenerator.make_dataset): dataset to flatten
            cols_to_flatten (list[str]): List of column names to flatten. This will put the \
                values of all the time steps as features.
            cols_keep_last_value (list[str], optional): List of column names that won't be flattened, \
                but for which only the last value will be kept. Defaults to [].
            label_cols_to_flatten (list[str], optional): List of label column names to flatten. \
                Defaults to [].

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Flattened inputs and labels

        Examples:
        >>> flat_inputs, flat_labels = w1.flatten_dataset(
        ...    w1.train, ["power"], ["date", "workday", "time_window"]
        ... )
        In this case, you will have 1*inputs_width + 3 features for the inputs and
        len(label_columns) * label_width
        """

        flat_inputs = []
        flat_labels = []

        # Warn if not all columns are selected for flattening or keeping
        columns_not_selected = [
            col
            for col in self.column_indices.keys()
            if col not in (cols_to_flatten + cols_keep_last_value)
        ]
        if columns_not_selected:
            print(
                "INFO: The following columns will be dropped when "
                f"flattening the inputs: {columns_not_selected}"
            )
        no_user_last_keep_value = False
        if not cols_keep_last_value:
            no_user_last_keep_value = True
            # if the user didn't specify any columns to keep, we keep the last value of the last column
            # so that the algorithm works. We will drop that column at the end
            cols_keep_last_value = [list(self.column_indices.keys())[-1]]

        # Precompute the indices for columns to flatten and keep last values
        input_flatten_indices = [self.column_indices[name] for name in cols_to_flatten]
        label_flatten_indices = [
            self.label_columns_indices[name] for name in label_cols_to_flatten
        ]

        keep_some_values_indices = [
            {
                "col_index_to_keep": self.column_indices[
                    dict_keep_some_values["col_name"]
                ],
                "value_indexes_to_keep": dict_keep_some_values["indexes_to_keep"],
            }
            for dict_keep_some_values in cols_keep_some_values
        ] + [
            {
                "col_index_to_keep": self.column_indices[col_name],
                "value_indexes_to_keep": [self.input_width - 1],
            }
            for col_name in cols_keep_last_value
        ]

        # Process each batch in the dataset
        for batch in data:
            inputs, labels = batch

            ### Flatten Inputs ###
            # inputs shape: (batch_size, inputs_width, num_features)
            if cols_to_flatten:
                # Gather and flatten columns across all time steps for each batch example
                items_to_flatten = tf.gather(
                    inputs, input_flatten_indices, axis=-1
                )  # shape: (batch_size, inputs_width, len(cols_to_flatten))
                items_to_flatten = tf.reshape(
                    items_to_flatten, [inputs.shape[0], -1]
                )  # Flatten across time steps per batch item
                # shape: (batch_size, inputs_width * len(cols_to_flatten))
            else:
                items_to_flatten = tf.zeros([inputs.shape[0], 0], dtype=tf.float32)

            # Initialize empty dataframe with good shape (batch_size, 0)
            items_to_keep_some_values = tf.zeros([inputs.shape[0], 0], dtype=tf.float32)
            # Gather the values of the columns in which we only keep some values
            for col_dict in keep_some_values_indices:
                col_index_to_keep = col_dict["col_index_to_keep"]
                value_indexes_to_keep = col_dict["value_indexes_to_keep"]
                items = tf.gather(
                    inputs[:, :, col_index_to_keep], value_indexes_to_keep, axis=1
                )  # shape: (batch_size, len(value_indexes_to_keep))
                items_to_keep_some_values = tf.concat(
                    [items_to_keep_some_values, items], axis=-1
                )

            # Concatenate flattened columns and the last values per example in the batch
            flat_input = tf.concat(
                [items_to_flatten, items_to_keep_some_values], axis=-1
            )  # shape: (batch_size, flattened_features + len(cols_keep_last_value))
            flat_inputs.append(
                flat_input.numpy()
            )  # Convert tensor to NumPy for DataFrame

            ### Flatten Labels ###
            # labels shape: (batch_size, time_steps, num_labels)

            # Gather and flatten label columns across all time steps for each batch example
            label_items_to_flatten = tf.gather(
                labels, label_flatten_indices, axis=-1
            )  # shape: (batch_size, time_steps, len(label_cols_to_flatten))
            flat_label = tf.reshape(
                label_items_to_flatten, [labels.shape[0], -1]
            )  # Flatten across time steps per batch item
            flat_labels.append(
                flat_label.numpy()
            )  # Convert tensor to NumPy for DataFrame compatibility

        # Combine the batch data into final flattened arrays
        flat_inputs = np.vstack(
            flat_inputs
        )  # Combine the flattened inputs across all batches
        flat_labels = np.vstack(
            flat_labels
        )  # Combine the flattened labels across all batches

        ### Generate Column Names ###
        input_column_names = (
            [f"{name}_{i}" for i in range(inputs.shape[1]) for name in cols_to_flatten]
            + [
                f"{dict_keep_some_values['col_name']}_{i}"
                for dict_keep_some_values in cols_keep_some_values
                for i in dict_keep_some_values["indexes_to_keep"]
            ]
            + cols_keep_last_value
        )
        label_column_names = [
            f"{name}_{i}"
            for i in range(labels.shape[1])
            for name in label_cols_to_flatten
        ]

        # Convert arrays to Pandas DataFrames
        flat_inputs_df = pd.DataFrame(flat_inputs, columns=input_column_names)
        if no_user_last_keep_value:
            # if the user didn't specify any columns to keep the last value from but we had to select
            # at least 1 for the algo to work. We must drop it now
            flat_inputs_df.drop(columns=cols_keep_last_value, inplace=True)
        flat_labels_df = pd.DataFrame(flat_labels, columns=label_column_names)

        return flat_inputs_df, flat_labels_df

    def get_sequence_stride(self, overlapping_windows: bool) -> int:
        """Get the sequence stride based on the overlapping_windows attribute."""
        if not overlapping_windows:
            return self.shift
        else:
            return 1

    def generate_val_from_train(self):
        full_train = self.make_dataset(
            self.train_df, shuffle=self.get_val_from_shuffled_train
        )
        self._train, self._val = self.split_dataset(full_train)

    @property
    def train(self) -> list[tuple]:
        if self.get_val_from_shuffled_train:
            _train = getattr(self, "_train", None)
            if not _train:
                self.generate_val_from_train()
                return self.train
            return _train
        else:
            return self.make_dataset(self.train_df, shuffle=False)

    @property
    def val(self) -> list[tuple]:
        if self.get_val_from_shuffled_train:
            _val = getattr(self, "_val", None)
            if not _val:
                self.generate_val_from_train()
                return self.val
            return _val
        else:
            return self.make_dataset(self.val_df, shuffle=False)

    @property
    def test(self):
        # TODO: do we really want to return a tuple here?
        return (self.make_dataset(self.test_df, shuffle=False),)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def convert_to_torch_dataset(
        self,
        dataset: list[tuple],
        cols_to_keep_as_features: list[str] | None = None,
        cols_to_keep_as_labels: list[str] | None = None,
    ) -> TFToTorchDataset:
        """Convert a TensorFlow dataset to a PyTorch dataset. The pytorch dataset doesn't have batches.

        Args:
            dataset (tf.data.Dataset): item from self.make_dataset()

        Returns:
            torch.utils.data.Dataset: a torch dataset. This doesn't have batches anymore. To get batches, use DataLoader.
        """
        if cols_to_keep_as_features is None:
            # we are going to keep all the columns
            keep_as_features_indices = list(self.column_indices.values())
        else:
            keep_as_features_indices = [
                self.column_indices[name] for name in cols_to_keep_as_features
            ]

        if cols_to_keep_as_labels is None:
            # we are going to keep all the columns
            keep_as_label_indices = list(self.label_columns_indices.values())
        else:
            keep_as_label_indices = [
                self.label_columns_indices[name] for name in cols_to_keep_as_labels
            ]
        return TFToTorchDataset(
            dataset, keep_as_features_indices, keep_as_label_indices
        )

    def plot(self, model=None, plot_col: str | None = None, max_subplots: int = 3):
        """Plot `max_sublots` sample examples, showing the inputs and labels
        for 1 column (`plot_col`).

        Args:
            model (_type_, optional): A model called to make prediction and add them \
                to the plots. The model is called by `predictions = model(inputs)`. \
                Defaults to None.
            plot_col (str, optional): Label column name to plot. If none, we will select \
                the first column of the labels. Defaults to None.
            max_subplots (int, optional): Number of samples to plot. Defaults to 3.
        """
        inputs, labels = self.example

        plt.figure(figsize=(12, 8))
        if not plot_col:
            # try to retrieve it from self.label_columns
            plot_col = (
                self.label_columns[0]
                if self.label_columns
                else self.train_df.columns[0]
            )
        plot_col_index = self.column_indices[plot_col]
        date_col_index = self.column_indices["date"]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            dates = pd.date_range(
                start=convert_date_from_int_to_datetime(inputs[n, :, :]).iloc[0],
                periods=self.total_window_size,
                freq="15min",
            )

            plt.plot(
                dates[self.input_indices],  # self.input_indices
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                dates[self.label_indices],  # self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    dates[self.label_indices],  # self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()
                plt.title("Random data from the training set")

        plt.xlabel(f"Time [{self.data_freq}]")

    def plot_train_val_split_selection(self):
        """Plot a 1D heatmap with the validation samples in red
        and the training samples in blue. The x-axis is the time steps.

        The goal of this plot is just to make sure that val and train are
        shuffled if that's what we want
        """
        # Get the first timestep of the labels for all the samples
        # Reshape is to put all timesteps 1 after the other (otherwise we would
        # have self.sequence_stride columns)
        all_train_label_dates = (
            self.convert_to_torch_dataset(self.train)
            .get_full_data()[1][:, : self.sequence_stride, 1]
            .numpy()
        ).reshape(-1)
        all_val_label_dates = (
            self.convert_to_torch_dataset(self.val)
            .get_full_data()[1][:, : self.sequence_stride, 1]
            .numpy()
        ).reshape(-1)

        df_all_train_label_dates = pd.DataFrame(all_train_label_dates, columns=["date"])
        df_all_train_label_dates["set"] = 0
        df_all_val_label_dates = pd.DataFrame(all_val_label_dates, columns=["date"])
        df_all_val_label_dates["set"] = 1

        df_all_dates = pd.concat([df_all_train_label_dates, df_all_val_label_dates])
        df_all_dates.sort_values("date", inplace=True)
        df_all_dates["date"] = pd.to_datetime(df_all_dates["date"], unit="s").dt.round(
            self.data_freq
        )

        resampling_freq = self.data_freq
        # We try to increase the resampling freq if possible, to have less data points to display
        if resampling_freq == "15min" and self.sequence_stride >= 4:
            resampling_freq = "1h"
        if resampling_freq == "5min" and self.sequence_stride >= 12:
            resampling_freq = "1h"
        df_all_dates = (
            df_all_dates.set_index("date").asfreq(resampling_freq).reset_index()
        )
        df_all_dates["set"] = df_all_dates["set"].fillna(2)

        fig = go.Figure()
        # Add a bicolor 1D heatmap.
        # 0 = training = blue, 1 = val = red, 2 = gap = grey
        fig.add_trace(
            go.Heatmap(
                z=[df_all_dates["set"]],
                x=df_all_dates["date"],
                y=["Set Values"],
                colorscale=[
                    [0, "rgba(0, 0, 255, 0.5)"],  # blue for training
                    [0.5, "rgba(255, 0, 0, 0.5)"],  # red for validation
                    [1, "rgba(200, 200, 200, 0.5)"],  # grey for gaps
                ],
                showscale=False,
            )
        )

        fig.update_layout(
            title="Training and Validation Split\n"
            "(Blue: Training, Red: Validation, Grey: Gap in the data)\n"
            "This graph should show that the validation set is randomly selected",
            xaxis_title="Time steps",
            yaxis_title="Samples",
        )

        fig.show()
