import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from .feature_engineering import reverse_feature_engineering

# source: https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TFToTorchDataset(Dataset):
    def __init__(
        self,
        tf_dataset,
        keep_last_value_indices: list[int],
        keep_as_label_indices: list[int],
    ):
        self.inputs = []
        self.labels = []

        # Iterate over TensorFlow batches and flatten to individual samples
        for batch in tf_dataset.as_numpy_iterator():
            inputs, labels = batch
            for i in range(len(inputs)):  # Unpack individual samples from the batch
                self.inputs.append(inputs[i][:, keep_last_value_indices])
                self.labels.append(labels[i][:, keep_as_label_indices])

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
        label_columns: list[str] | None = None,
        overlapping_windows: bool = False,
        batch_size: int = 64,
    ):
        """_summary_

        Args:
            input_width (int): Number of time steps to include in the input window.
            label_width (int): Number of time steps to include in the label window.
            shift (int): Number of time steps to shift the start of the label window relative to the start of the input window. Usually equal to input_width if you don't want to overlap input and label windows.
            train_df (pd.DataFrame): The training dataset, in the "FeaturedEngineeredSchema" format.
            test_df (pd.DataFrame | None, optional): The test dataset, in the "FeaturedEngineeredSchema" format. Defaults to None.
            val_df (pd.DataFrame | None, optional): The validation dataset, in the "FeaturedEngineeredSchema" format. Defaults to None.
            label_columns (list[str] | None, optional): List of column names to use as labels. Defaults to None.
            overlapping_windows (bool, optional): Whether to have overlapping label windows in the data. If the shift is greater than the label width, inputs will overlap (which is fine!)Defaults to False.
            batch_size (int, optional): Defaults to 32. For now, set this to 1 if you want to use flatten_dataset.
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.data_freq = self._get_data_frequency(train_df)

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

    def _get_data_frequency(self, df):
        """Get the data frequency from the DataFrame."""
        if isinstance(df["date"].iloc[0], pd.Timestamp):
            return pd.infer_freq(df["date"])
        else:
            return pd.infer_freq(pd.to_datetime(df["date"], unit="s"))

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
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

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(  # type: ignore
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=False,
            batch_size=self.batch_size,
        )

        ds = ds.map(self.split_window)

        return ds

    def flatten_dataset_old(
        self,
        data,
        cols_to_flatten: list[str],
        cols_keep_last_value: list[str] = [],
        label_cols_to_flatten: list[str] = [],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """DEPRECIATED - Use new function that is much faster
        For each example, flattens the 2D input data into a 1D array.


        Args:
            data (output of WindowGenerator.make_dataset): dataset to flatten
            cols_to_flatten (list[str]): List of column names to flatten. This will put the values
                of all the time steps as features.
            cols_keep_last_value (list[str], optional): List of column names that won't be flattened, but for which
                only the last value will be kept. Defaults to [].
            label_cols_to_flatten (list[str], optional): List of label column names to flatten.
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

        # Add a warning if not all of the columns are selected
        columns_not_selected = [
            col
            for col in self.column_indices.keys()
            if col not in (cols_to_flatten + cols_keep_last_value)
        ]
        if columns_not_selected:
            print(
                f"WARNING: The following columns will be dropped when flattening: {columns_not_selected}"
            )

        if self.batch_size != 1:
            # TODO: make it work for batch size > 1. Need to return batches instead of just one item
            raise ValueError("flatten_dataset only works with batch size of 1")

        for batch in data:
            inputs, labels = batch

            ### Inputs
            # TODO: works only for batchsize = 1
            input_batch_item = inputs[0]

            items_to_flatten = np.concatenate(
                [
                    input_batch_item[:, self.column_indices[name]]
                    for name in cols_to_flatten
                ],
                axis=-1,
            )
            items_to_keep_last_value = [
                input_batch_item[-1, self.column_indices[name]]
                for name in cols_keep_last_value
            ]

            flat_inputs.append(
                np.concatenate([items_to_flatten, items_to_keep_last_value])
            )

            ### Outputs
            # TODO: works only for batchsize = 1
            labels_batch_item = labels[0]
            if not label_cols_to_flatten:
                label_cols_to_flatten = self.label_columns
                print(
                    f"WARNING: No label columns to flatten. Using all label columns ({label_cols_to_flatten})"
                )
            label_items_to_flatten = [
                labels_batch_item[:, self.label_columns_indices[name]]
                for name in label_cols_to_flatten
            ]
            flat_labels.append(np.concatenate([*label_items_to_flatten]))

        input_column_names = [
            f"{name}_{i}" for name in cols_to_flatten for i in self.input_indices
        ] + cols_keep_last_value
        label_column_names = [
            f"{name}_{i}" for name in label_cols_to_flatten for i in self.label_indices
        ]
        flat_inputs = pd.DataFrame(flat_inputs, columns=input_column_names)
        flat_labels = pd.DataFrame(flat_labels, columns=label_column_names)
        return flat_inputs, flat_labels

    def flatten_dataset(
        self,
        data,
        cols_to_flatten: list[str],
        cols_keep_last_value: list[str] = [],
        label_cols_to_flatten: list[str] = [],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """For each example, flattens the 2D input data into a 1D array.
        This version is more efficient than the previous one and works with batch size > 1.
        Use larger batch size to speed up the computation.

        Args:
            data (output of WindowGenerator.make_dataset): dataset to flatten
            cols_to_flatten (list[str]): List of column names to flatten. This will put the values
                of all the time steps as features.
            cols_keep_last_value (list[str], optional): List of column names that won't be flattened, but for which
                only the last value will be kept. Defaults to [].
            label_cols_to_flatten (list[str], optional): List of label column names to flatten.
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
                f"WARNING: The following columns will be dropped when flattening: {columns_not_selected}"
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
        keep_last_value_indices = [
            self.column_indices[name] for name in cols_keep_last_value
        ]

        # Process each batch in the dataset
        for batch in data:
            inputs, labels = batch

            ### Flatten Inputs ###
            # inputs shape: (batch_size, inputs_width, num_features)

            # Gather and flatten columns across all time steps for each batch example
            items_to_flatten = tf.gather(
                inputs, input_flatten_indices, axis=-1
            )  # shape: (batch_size, inputs_width, len(cols_to_flatten))
            items_to_flatten = tf.reshape(
                items_to_flatten, [inputs.shape[0], -1]
            )  # Flatten across time steps per batch item
            # shape: (batch_size, inputs_width * len(cols_to_flatten))

            # Gather last value for the specified columns
            items_to_keep_last_value = tf.gather(
                inputs[:, -1, :], keep_last_value_indices, axis=-1
            )  # shape: (batch_size, len(cols_keep_last_value))

            # Concatenate flattened columns and the last values per example in the batch
            flat_input = tf.concat(
                [items_to_flatten, items_to_keep_last_value], axis=-1
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
        input_column_names = [
            f"{name}_{i}" for name in cols_to_flatten for i in range(inputs.shape[1])
        ] + cols_keep_last_value
        label_column_names = [
            f"{name}_{i}"
            for name in label_cols_to_flatten
            for i in range(labels.shape[1])
        ]

        # Convert lists to Pandas DataFrames
        flat_inputs_df = pd.DataFrame(flat_inputs, columns=input_column_names)
        if no_user_last_keep_value:
            # if the user didn't specify any columns to keep the last value from but we had to select
            # at least 1 for the algo to work. We must drop it now
            flat_inputs_df.drop(columns=cols_keep_last_value, inplace=True)
        flat_labels_df = pd.DataFrame(flat_labels, columns=label_column_names)

        return flat_inputs_df, flat_labels_df

    def get_sequence_stride(self, overlapping_windows: bool):
        """Get the sequence stride based on the overlapping_windows attribute."""
        if not overlapping_windows:
            return self.shift
        else:
            return 1

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

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
        dataset: tf.data.Dataset,
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

    def plot(self, model=None, plot_col=None, max_subplots=3):
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
                start=reverse_feature_engineering(inputs[n, :, :])["date"].iloc[0],
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
