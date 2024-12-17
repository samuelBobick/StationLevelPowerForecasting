from typing import Literal

import numpy as np
import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
import torch
import torch.nn as nn
from slrp_ev_data.window_generator import TFToTorchDataset
from slrp_ev_ts_forecasting.models.torch_base import TorchBaseModel
from torch.nn.utils.parametrizations import weight_norm


class TCN(TorchBaseModel):
    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        hidden_size: int = 16,
        num_layers: int | Literal["auto"] = "auto",
        kernel_size: int = 3,
        dropout: float = default_parameters.DROPOUT,
        activation=nn.ReLU(),
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 3,
        epochs: int = default_parameters.EPOCHS,
        number_of_initial_models: int = default_parameters.NUMBER_OF_INITIAL_MODELS,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        batch_size: int = default_parameters.BATCH_SIZE,
        use_decoder: bool = True,
        error_metric: default_parameters.TypeErrorMetric = default_parameters.ERROR_METRIC,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
    ):
        """
        Initialize the TCN model with the given parameters.

        Args:
            x_dim (int): Input dimension.
            lookahead (int): Number of steps to predict ahead.
            alpha (int): Alpha parameter for the model.
            hidden_size (int): Number of hidden units in the TCN.
            num_layers (int): Number of layers.
            kernel_size (int): Kernel size for the TCN.
            dropout (float): Dropout rate for the TCN.
            activation (nn.Module): Activation function to use in the TCN.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            lr_threshold (float): Threshold for learning rate scheduler.
            scheduler_patience (int): Patience for learning rate scheduler.
            epochs (int): Number of epochs to train the model.
            number_of_initial_models (int): Number of initial models to train.
            time_mode (Literal["window", "cyclical"]): Time mode for the model.
            batch_size (int): Batch size for training.
            use_decoder (bool): Whether to use a decoder in the model. \
                The decoder is an additional layer that allows reshaping \
                the output to the output_length. If no decoder is used, \
                the output length is equal to the input length, \
                so there will be some of the input in what the model \
                is trying to predict. You then have to use self.first_prediction_index \
                to get the index from which the predictions start.
            error_metric (default_parameters.TypeErrorMetric): Error metric to \
                use for training.
            get_val_data_from_shuffled_train (bool): Whether to get the \
                validation data from the shuffled train data. This can help \
                improving the algorithm's performance since there will more \
                recent data in the training set (otherwise, the most recent data \
                is in the val and test sets)
        """
        # TCN-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.activation = activation
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilatation_base = 2
        self.use_decoder = use_decoder

        if num_layers == "auto":
            self.num_layers = self.get_num_layers()
        else:
            self.num_layers = num_layers
            if self.num_layers < self.get_num_layers():
                print(
                    "WARNING: num_layers is less than the minimum number of layers required for full history coverage"
                )

        # Initialize the BaseModel with relevant parameters
        super().__init__(
            epochs=epochs,
            number_of_initial_models=number_of_initial_models,
            batch_size=batch_size,
            model_str_name=self.model_str_name,
            alpha=alpha,
            initial_learning_rate=initial_learning_rate,
            lr_threshold=lr_threshold,
            scheduler_patience=scheduler_patience,
            error_metric=error_metric,
            x_dim=x_dim,
            lookahead=lookahead,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
        )

        # Other parameters
        self.alpha = alpha
        self.time_mode = time_mode

        # Determine input size based on time_mode
        self.nr_input_channels = self._determine_input_size()

    @property
    def model_str_name(self):
        return (
            f"TCN_hidSize{self.hidden_size}"
            + f"_layers{self.num_layers}"
            + f"_kernelSize{self.kernel_size}"
            + f"_dropout{self.dropout}"
        )

    def get_num_layers(self) -> int:
        """Computes the minimum number of layers required for full history coverage
        (meaning that the prediction is based on all the past data).
        That formula is given here: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
        """
        le = self.x_dim
        k = self.kernel_size
        b = self.dilatation_base
        return int(
            np.ceil(np.log(((le - 1) * (b - 1)) / (2 * (k - 1)) + 1) / np.log(b))
        )

    def initialize_model(self) -> None:
        """Initializes the TCN model based on the current configuration."""
        self.model = TCN_model(
            nr_input_channels=self.nr_input_channels,
            input_length=self.x_dim,
            output_length=self.lookahead,
            # since we want to predict only 1 variable (power), we set the last channel to 1.
            # the other channels have the hidden_size.
            num_channels=[self.hidden_size] * (self.num_layers - 1) + [1],
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            activation=self.activation,
            use_decoder=self.use_decoder,
        )
        self.model.to(default_parameters.DEVICE)

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        if self.time_mode == "window":
            return (
                1 + 6 + 1
            )  # Example: 1 for power, 6 for time one-hot encoding, 1 for workday
        elif self.time_mode == "cyclical":
            return (
                1 + 6
            )  # Example: 1 for power, 6 for sin/cos encoding, (day, week, year)
        else:
            raise ValueError(f"Invalid time_mode: {self.time_mode}")

    def get_dataset(
        self,
        df: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, pd.DataFrame]:
        """Generates the dataset and features based on the input DataFrame."""
        label_width = self.lookahead
        if not self.use_decoder:
            # To have an output of the same length as the input
            label_width = self.x_dim
        if df is not None:
            df = df.copy()
            df_padded = self.pad_with_seen_data(
                df, number_of_timesteps_to_pad=self.x_dim
            )
        else:
            df_padded = None

        W, window_data = self.get_window_data(
            df_padded, self.x_dim, label_width, overlapping_windows, data_type
        )

        cols_to_keep_as_features = ["power"]
        if self.time_mode == "cyclical":
            cols_to_keep_as_features += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ]
        elif self.time_mode == "window":
            cols_to_keep_as_features += ["time_window", "workday"]

        dataset = W.convert_to_torch_dataset(
            window_data, cols_to_keep_as_features, cols_to_keep_as_labels=["power"]
        )

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                window_data, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            # x_dates, y_dates = W.convert_to_torch_dataset(
            #     window_data,
            #     cols_to_keep_as_features=["date"],
            #     cols_to_keep_as_labels=["date"],
            # ).get_full_data()
            return dataset, y_dates
        else:
            return dataset

    @property
    def first_prediction_index(self) -> int:
        return -self.lookahead


# TCN model implementation, source:
# https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script
# explanation of TCN can be found here:
# https://arxiv.org/pdf/1803.01271.pdf
# https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        # Each block consists of two dilated causal convolutions.
        # In between them, batch normalization, Relu activation and dropout are applied.
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        # we only add left padding
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        self.net = nn.Sequential(
            self.pad,
            self.conv1,
            self.relu,
            self.dropout,
            self.pad,
            self.conv2,
        )
        if n_outputs != 1:
            self.net.append(self.relu)
        self.net.append(self.dropout)
        # 1x1 convolution that will be used for the first and last blocks of the full network,
        # to match the number of input channels to the number of output channels
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(
        self, num_inputs, num_channels, dilatation_base, kernel_size=2, dropout=0.2
    ):
        super(TemporalConvNet, self).__init__()

        # Comes from # https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
        # 'Generally speaking, for a receptive field with no holes, the kernel size
        # k has to be at least as big as the dilation base b.'
        assert (
            kernel_size >= dilatation_base
        ), f"kernel_size {kernel_size} must be greater than the dilatation base {dilatation_base}"
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilatation_size = dilatation_base**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilatation_size,
                    padding=(kernel_size - 1) * dilatation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_model(nn.Module):
    def __init__(
        self,
        nr_input_channels,
        input_length,
        output_length,
        num_channels,
        activation,
        dilatation_base=2,
        kernel_size=2,
        dropout=0.2,
        use_decoder=True,
    ):
        super(TCN_model, self).__init__()
        self.tcn = TemporalConvNet(
            nr_input_channels,
            num_channels,
            dilatation_base,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.output_length = output_length

        # define decoder
        self.use_decoder = use_decoder
        if self.use_decoder:
            self.activation = activation
            self.fc_1 = nn.Linear(input_length, 128)  # fully connected layer
            self.fc = nn.Linear(128, output_length)  # output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """perform a forward pass on the TCN model

        Args:
            x: size should be [batch_size, sequence_length, nr_input_channels]

        Returns:
            prediction of the model
        """
        # convert x of size [batch_size, sequence_length, nr_input_channels] to
        # [batch_size, nr_input_channels, sequence_length]
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # return self.linear(out[:, :, -1])

        # convert to length [batch_size, sequence_length, nr_output_channels(=1)]
        out = out.transpose(1, 2)

        # # remove last dimension (which is 1)
        if self.use_decoder:
            out = out.squeeze()
            out = self.activation(self.fc_1(out))
            out = self.fc(out)
            out = nn.functional.softplus(out)  # ensuring non-negative output
        return out
