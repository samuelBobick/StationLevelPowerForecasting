from typing import Literal

import pandas as pd
import torch
import torch.nn as nn
from slrp_ev_data.feature_engineering import one_hot_encoding
from slrp_ev_data.window_generator import WindowGenerator
from torch.utils.data import Dataset

import slrp_ev_ts_forecasting.default_parameters as default_parameters
from slrp_ev_ts_forecasting.torch_base import TorchBaseModel

# Suppress all warnings
# warnings.filterwarnings("ignore")


class FFNN(TorchBaseModel):
    def __init__(
        self,
        x_dim: int = default_parameters.X_DIM,
        lookahead: int = default_parameters.LOOKAHEAD,
        alpha: int = default_parameters.ALPHA,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
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
        """TODO"""

        # FFNN-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_hidden_layers = num_hidden_layers
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
        )

        # Determine input size based on time_mode
        self.input_size = self._determine_input_size()

    @property
    def model_str_name(self):
        return f"FFNN_hidSize{self.hidden_size}_layers{self.num_hidden_layers}"

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        if self.time_mode == "window":
            return self.x_dim + 6 + 1  # 6 for time one-hot encoding, 1 for workday
        elif self.time_mode == "cyclical":
            return self.x_dim + 6  #  6 for sin/cos encoding (day, week, year)
        else:
            raise ValueError(f"Invalid time_mode: {self.time_mode}")

    def initialize_model(self) -> None:
        """Initializes the FFNN model based on the current configuration."""
        self.model = FFNN_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
        )
        self.model.to(default_parameters.DEVICE)

    def get_dataset(
        self,
        df: pd.DataFrame,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> Dataset | tuple[Dataset, torch.Tensor]:
        W = WindowGenerator(
            input_width=self.x_dim,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
        )
        cols_keep_last_value = []
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
            cols_keep_last_value += ["time_window", "workday"]

        flat_inputs, flat_labels = W.flatten_dataset(
            W.train,
            cols_to_flatten=["power"],
            cols_keep_last_value=cols_keep_last_value,
            label_cols_to_flatten=["power"],
        )
        if self.time_mode == "window":
            flat_inputs = one_hot_encoding(flat_inputs, ["time_window"])

        dataset = TensorDataset(flat_inputs, flat_labels)

        if return_y_date:
            x_dates, y_dates = W.convert_to_torch_dataset(
                W.train,
                cols_to_keep_as_features=["date"],
                cols_to_keep_as_labels=["date"],
            ).get_full_data()
            return dataset, y_dates
        else:
            return dataset


class FFNN_model(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden_layers, activation
    ):
        super(FFNN_model, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)]
        )
        self.activation = activation
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.fc_out(x)
        # Add a continuous activation function to unsure results are positive
        x = nn.functional.softplus(x)
        return x


class TensorDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = torch.tensor(
            x.values, dtype=torch.float32, device=default_parameters.DEVICE
        )
        self.y = torch.tensor(
            y.values, dtype=torch.float32, device=default_parameters.DEVICE
        )
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def get_full_data(self):
        """
        Returns:
        - self.x: Full dataset input as a PyTorch tensor
        - self.y: Full dataset labels as a PyTorch tensor
        """
        # Convert lists of arrays to PyTorch tensors
        return self.x, self.y
