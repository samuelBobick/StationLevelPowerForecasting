from typing import Literal

import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
import torch
import torch.nn as nn
from slrp_ev_ts_forecasting.models.torch_base import TorchBaseModel
from torch.utils.data import Dataset

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
        dropout: float = 0.2,
        batch_norm: bool = False,
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 5,
        epochs: int = default_parameters.EPOCHS,
        number_of_initial_models: int = default_parameters.NUMBER_OF_INITIAL_MODELS,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        batch_size: int = default_parameters.BATCH_SIZE,
        error_metric: default_parameters.TypeErrorMetric = default_parameters.ERROR_METRIC,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        scaling_mode: default_parameters.TypeScalingMode = default_parameters.SCALING_MODE,
        scaling_parameters: tuple | pd.DataFrame | None = None,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
        number_of_artificial_datasets: int = default_parameters.NUMBER_OF_ARTIFICIAL_DATASETS,
        random_start_time: bool = default_parameters.RANDOM_START_TIME,
        shuffle_power_profiles: bool = default_parameters.SHUFFLE_POWER_PROFILES,
        random_power_profile_shapes: bool = default_parameters.RANDOM_POWER_PROFILE_SHAPES,
        random_user_needs: bool = default_parameters.RANDOM_USER_NEEDS,
        random_choices: bool = default_parameters.RANDOM_CHOICES,
        add_number_of_evses_available: bool = default_parameters.ADD_NUMBER_OF_EVSES_AVAILABLE,
    ):
        """TODO"""

        # FFNN-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Regression specific parameters
        self.optimize_lags = optimize_lags
        self.session_based_mode = session_based_mode
        self.peak_prediction = peak_prediction
        self.add_number_of_sessions = add_number_of_sessions
        self.add_fraction_of_regular_sessions = add_fraction_of_regular_sessions
        self.use_all_active_sessions = use_all_active_sessions

        # Other parameters
        self.alpha = alpha
        self.time_mode: Literal["window", "cyclical"] = time_mode
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
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
            optimize_lags=optimize_lags,
            time_mode=time_mode,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
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
            f"FFNN_hidSize{self.hidden_size}_layers{self.num_hidden_layers}"
            + ("_lagsOpti" if self.optimize_lags else "")
            + ("Short" if self.optimize_lags == "short_opt" else "")
            + ("Long" if self.optimize_lags == "long_opt" else "")
            + f"_dropout{self.dropout}"
            + ("_withBatchNorm" if self.batch_norm else "")
            + (
                "_SessionBased"
                + ("_PeakPrediction" if self.peak_prediction else "")
                + ("_WithNbSessions" if self.add_number_of_sessions else "")
                + ("_WithFracReg" if self.add_fraction_of_regular_sessions else "")
                + ("_WithAllActiveSessions" if self.use_all_active_sessions else "")
                if self.session_based_mode
                else ""
            )
        )

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        input_size = (
            self.x_dim
            + int(self.add_number_of_evses_available)
            + len(self.list_workday_column_names)
        )
        if self.session_based_mode:
            input_size += (
                self.lookahead
                + int(self.add_number_of_sessions)
                + int(self.add_fraction_of_regular_sessions)
            )

        if self.time_mode == "window":
            return input_size + 6  # 6 for time one-hot encoding
        elif self.time_mode == "cyclical":
            return input_size + 6  #  6 for sin/cos encoding (day, week, year)
        else:
            raise ValueError(f"Invalid time_mode: {self.time_mode}")

    def initialize_model(self) -> None:
        """Initializes the FFNN model based on the current configuration."""
        output_size = self.output_size
        if self.peak_prediction:
            output_size = 1
        self.model = FFNN_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            force_positive_output=self.scaling_mode
            in ["normalize", "rolling_normalize"],
        )
        self.model.to(default_parameters.DEVICE)

    def get_dataset(
        self,
        df: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> Dataset | tuple[Dataset, pd.DataFrame]:

        samples = self.get_X_y(
            df=df,
            time_mode=self.time_mode,
            data_type=data_type,
            return_y_date=return_y_date,
            overlapping_windows=overlapping_windows,
            multi_model_mode=False,
        )

        if len(samples) == 3:
            flat_inputs, flat_labels, y_dates = samples
        else:
            flat_inputs, flat_labels = samples

        dataset = TensorDataset(flat_inputs, flat_labels)
        if return_y_date:
            return dataset, y_dates  # type: ignore
        else:
            return dataset


class FFNN_model(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_hidden_layers,
        activation,
        dropout,
        batch_norm,
        force_positive_output=True,
    ):
        super(FFNN_model, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)]
        )
        self.activation = activation
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.force_positive_output = force_positive_output

    def forward(self, x) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.batch_norm and x.size(0) > 1:
                # We don't want to apply batch norm if the batch size
                # is 1. This can happen with the last batch of the dataset
                x = self.batch_norm(x)
            x = self.dropout(x)

        x = self.fc_out(x)
        if self.force_positive_output:
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
