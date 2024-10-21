from typing import Literal

import pandas as pd
import torch
import torch.nn as nn
from slrp_ev_data.window_generator import TFToTorchDataset, WindowGenerator

import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_ts_forecasting.torch_base import TorchBaseModel


class LSTM(TorchBaseModel):
    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        hidden_size: int = 32,
        num_lstm_layers: int = 1,
        activation=nn.ReLU(),
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 5,
        epochs: int = default_parameters.EPOCHS,
        number_of_initial_models: int = default_parameters.NUMBER_OF_INITIAL_MODELS,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        batch_size: int = default_parameters.BATCH_SIZE,
        error_metric: default_parameters.TypeErrorMetric = default_parameters.ERROR_METRIC,
    ):
        """
        Initialize the LSTM model with the given parameters.

        Args:
            x_dim (int): Input dimension.
            lookahead (int): Number of steps to predict ahead.
            alpha (int): Alpha parameter for the model.
            hidden_size (int): Number of hidden units in the LSTM.
            num_lstm_layers (int): Number of LSTM layers.
            activation (nn.Module): Activation function to use in the LSTM.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            lr_threshold (float): Threshold for learning rate scheduler.
            scheduler_patience (int): Patience for learning rate scheduler.
            epochs (int): Number of epochs to train the model.
            number_of_initial_models (int): Number of initial models to train.
            time_mode (Literal["window", "cyclical"]): Time mode for the model.
            batch_size (int): Batch size for training.
        """

        # LSTM-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_lstm_layers = num_lstm_layers
        self.activation = activation

        # Other parameters
        self.alpha = alpha
        self.time_mode = time_mode

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
        )

        # Determine input size based on time_mode
        self.input_size = self._determine_input_size()

    @property
    def model_str_name(self):
        return f"LSTM_hidSize{self.hidden_size}_lstmLayers{self.num_lstm_layers}"

    def initialize_model(self) -> None:
        """Initializes the LSTM model based on the current configuration."""
        self.model = LSTM_model(
            output_size=self.output_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            activation=self.activation,
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
        df: pd.DataFrame,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, torch.Tensor]:
        """Generates the dataset and features based on the input DataFrame."""
        df = self.pad_with_seen_data(df, number_of_timesteps_to_pad=self.x_dim)

        W = WindowGenerator(
            input_width=self.x_dim,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
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
            W.train, cols_to_keep_as_features, cols_to_keep_as_labels=["power"]
        )

        if return_y_date:
            x_dates, y_dates = W.convert_to_torch_dataset(
                W.train,
                cols_to_keep_as_features=["date"],
                cols_to_keep_as_labels=["date"],
            ).get_full_data()
            return dataset, y_dates
        else:
            return dataset


class LSTM_model(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, activation):
        super(LSTM_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected layer
        self.fc = nn.Linear(128, output_size)  # output layer

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x: Input of shape [batch_size, seq_length, input_size], where seq_length
               is the number of time steps, and input_size is the number of features.

        Returns:
            out: Output of the network
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )  # internal state

        # Propagate input through LSTM
        output, (hn, _) = self.lstm(x, (h_0, c_0))

        # hn is of size [num_layers, batch_size, hidden_size]
        hn = hn[-1]  # use the last layer's output
        out = self.activation(hn)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc(out)
        out = nn.functional.softplus(out)  # ensuring non-negative output
        return out
