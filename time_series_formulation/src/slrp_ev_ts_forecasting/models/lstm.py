from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
import torch
import torch.nn as nn
from slrp_ev_ts_forecasting.models.torch_base import TorchBaseModel


class LSTM(TorchBaseModel):
    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        hidden_size: int = 32,
        num_lstm_layers: int = 1,
        activation=nn.ReLU(),
        dropout: float = default_parameters.DROPOUT,
        batch_norm: bool = default_parameters.BATCH_NORM,
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 5,
        epochs: int = default_parameters.EPOCHS,
        number_of_initial_models: int = default_parameters.NUMBER_OF_INITIAL_MODELS,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        batch_size: int = default_parameters.BATCH_SIZE,
        error_metric: default_parameters.TypeErrorMetric = default_parameters.ERROR_METRIC,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        scaling_mode: default_parameters.TypeScalingMode = default_parameters.SCALING_MODE,
        scaling_parameters: tuple | pd.DataFrame | None = None,
        number_of_artificial_datasets: int = default_parameters.NUMBER_OF_ARTIFICIAL_DATASETS,
        random_start_time: bool = default_parameters.RANDOM_START_TIME,
        shuffle_power_profiles: bool = default_parameters.SHUFFLE_POWER_PROFILES,
        random_power_profile_shapes: bool = default_parameters.RANDOM_POWER_PROFILE_SHAPES,
        random_user_needs: bool = default_parameters.RANDOM_USER_NEEDS,
        random_choices: bool = default_parameters.RANDOM_CHOICES,
        add_number_of_evses_available: bool = default_parameters.ADD_NUMBER_OF_EVSES_AVAILABLE,
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
            error_metric (default_parameters.TypeErrorMetric): Error metric to \
                use for training.
            get_val_data_from_shuffled_train (bool): Whether to get the \
                validation data from the shuffled train data. This can help \
                improving the algorithm's performance since there will more \
                recent data in the training set (otherwise, the most recent data \
                is in the val and test sets)
        """

        # LSTM-specific parameters
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_lstm_layers = num_lstm_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Other parameters
        self.alpha = alpha
        self.time_mode = time_mode
        self.add_number_of_evses_available = add_number_of_evses_available

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
            optimize_lags=None,
            time_mode=time_mode,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
            session_based_mode=False,
            peak_prediction=False,
            add_number_of_sessions=False,
            add_fraction_of_regular_sessions=False,
            use_all_active_sessions=False,
            number_of_artificial_datasets=number_of_artificial_datasets,
            random_start_time=random_start_time,
            shuffle_power_profiles=shuffle_power_profiles,
            random_power_profile_shapes=random_power_profile_shapes,
            random_user_needs=random_user_needs,
            random_choices=random_choices,
            add_number_of_evses_available=add_number_of_evses_available,
        )

        # Determine input size based on time_mode
        self.input_size = self._determine_input_size()

    @property
    def model_str_name(self):
        return (
            f"LSTM_hidSize{self.hidden_size}"
            + f"_lstmLayers{self.num_lstm_layers}"
            + f"_dropout{self.dropout}"
            + ("_withBatchNorm" if self.batch_norm else "")
        )

    def initialize_model(self) -> None:
        """Initializes the LSTM model based on the current configuration."""
        self.model = LSTM_model(
            output_size=self.output_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        )
        self.model.to(default_parameters.DEVICE)

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        input_size = int(self.add_number_of_evses_available)
        if self.time_mode == "window":
            return (
                input_size + 1 + 6 + 1
            )  # Example: 1 for power, 6 for time one-hot encoding, 1 for workday
        elif self.time_mode == "cyclical":
            return (
                input_size + 1 + 6
            )  # Example: 1 for power, 6 for sin/cos encoding, (day, week, year)
        else:
            raise ValueError(f"Invalid time_mode: {self.time_mode}")


class LSTM_model(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        hidden_size,
        num_layers,
        activation,
        dropout,
        batch_norm,
    ):
        super(LSTM_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
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
        # hn is now of size [batch_size, hidden_size]
        if (self.batch_norm is not None) and (hn.size(0) > 1):
            hn = self.batch_norm(hn)
        hn = self.dropout(hn)
        out = self.activation(hn)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc(out)
        out = nn.functional.softplus(out)  # ensuring non-negative output
        return out
