from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from neural_network_base import AsymmetricRMSELoss, BaseModel
from slrp_ev_data.window_generator import TFToTorchDataset, WindowGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm


class LSTM(BaseModel):
    def __init__(
        self,
        x_dim: int = 16,
        lookahead: int = 16,
        alpha: int = 2,
        hidden_size: int = 64,
        num_lstm_layers: int = 1,
        activation=nn.ReLU(),
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 3,
        epochs: int = 100,
        number_of_initial_models: int = 5,
        time_mode: Literal["window", "cyclical"] = "cyclical",
        batch_size: int = 64,
    ):
        # Initialize the BaseModel with relevant parameters
        super().__init__(
            initial_learning_rate=initial_learning_rate,
            scheduler_patience=scheduler_patience,
            epochs=epochs,
            model_str_name="lstm",
            lr_threshold=lr_threshold,
        )

        # LSTM-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_lstm_layers = num_lstm_layers
        self.activation = activation

        # Other parameters
        self.batch_size = batch_size
        self.number_of_initial_models = number_of_initial_models
        self.alpha = alpha
        self.time_mode = time_mode

        # Determine input size based on time_mode
        self.input_size = self._determine_input_size()

        # Initialize model, optimizer, and scheduler
        self.initialize_model()
        self.initialize_optimizer_scheduler()

    def initialize_model(self) -> None:
        """Initializes the LSTM model based on the current configuration."""
        self.model = LSTM_model(
            output_size=self.output_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            activation=self.activation,
        )

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        if self.time_mode == "window":
            return (
                1 + 6 + 1
            )  # Example: 1 for power, 6 for time one-hot encoding, 1 for workday
        elif self.time_mode == "cyclical":
            return (
                1 + 1 + 4
            )  # Example: 1 for power, 1 for workday, 4 for sin/cos encoding
        else:
            raise ValueError(f"Invalid time_mode: {self.time_mode}")

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> None:
        """Find best model (out of number_of_initial_models) and train it on the entire dataset."""
        train_loader = self.get_dataloader(
            train, shuffle=True, overlapping_windows=True
        )
        val_loader = self.get_dataloader(val, shuffle=False, overlapping_windows=False)

        self.add_model_to_board(train_loader)

        self.best_vloss = np.inf
        # Train number_of_initial_models models and save the best one
        for i in (pbar := tqdm(range(self.number_of_initial_models))):
            pbar.set_description(
                f"Training Initial Model {i + 1}/{self.number_of_initial_models}"
            )

            # Re-initialize the model for each initial model training
            self.initialize_model()
            self.initialize_optimizer_scheduler()

            self.fit_one_model(
                train_loader, val_loader, epochs=3, best_vloss=self.best_vloss
            )

        # At this point, we have started training number_of_initial_models and we saved the best one
        # We can load the checkpoint of the best one and resume training
        current_model_epoch = self.load_checkpoint()
        self.fit_one_model(
            train_loader,
            val_loader,
            epochs=self.epochs - current_model_epoch,
            start_epoch=current_model_epoch + 1,
            writer=self.best_model_writer,
            best_vloss=self.best_vloss,
        )

    def predict(
        self, test: pd.DataFrame
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Given a pandas DataFrame test, returns error metrics and list of predictions."""
        dataset, y_dates = self.get_dataset(
            test, return_y_date=True, overlapping_windows=False
        )  # type:ignore
        dataset: TFToTorchDataset = dataset
        y_dates: pd.DataFrame = y_dates
        X_test_tensor, y_test_tensor = dataset.get_full_data()
        y_test_tensor = y_test_tensor.squeeze()

        # Load model from the checkpoint
        self.load_checkpoint()

        self.model.eval()
        y_pred_test = self.model(X_test_tensor).detach().numpy().squeeze()
        y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32)

        rmse = torch.sqrt(self.criterion(y_test_tensor, y_pred_test_tensor)).item()

        # Compute weighted RMSE using a custom asymmetric loss function
        weighted_criterion = AsymmetricRMSELoss(alpha=self.alpha)
        wrmse = torch.sqrt(weighted_criterion(y_test_tensor, y_pred_test_tensor)).item()

        # Flatten the lists to 1D
        y_pred_test_flat = y_pred_test.flatten()
        forecast_dates = y_dates.to_numpy().flatten()

        return rmse, wrmse, y_pred_test_flat, forecast_dates

    def get_dataset(
        self,
        df: pd.DataFrame,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, pd.DataFrame]:
        """Generates the dataset and features based on the input DataFrame."""
        W = WindowGenerator(
            input_width=self.x_dim,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
        )

        cols_to_keep_as_features = ["power", "workday"]
        if self.time_mode == "cyclical":
            cols_to_keep_as_features += ["Day sin", "Day cos", "Year sin", "Year cos"]
        elif self.time_mode == "window":
            cols_to_keep_as_features += ["time_window"]

        dataset = W.convert_to_torch_dataset(
            W.train, cols_to_keep_as_features, cols_to_keep_as_labels=["power"]
        )

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                W.train, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            return dataset, y_dates
        else:
            return dataset

    def get_dataloader(
        self,
        df: pd.DataFrame,
        shuffle: bool = False,
        overlapping_windows: bool = False,
    ) -> DataLoader:
        """Given the dataset, returns a DataLoader object."""
        dataset: TFToTorchDataset = self.get_dataset(
            df, overlapping_windows=overlapping_windows
        )  # type: ignore
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)


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
        # out = nn.functional.softplus(out)  # ensuring non-negative output
        return out
