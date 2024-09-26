from typing import Literal

import numpy as np
import pandas as pd
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
from neural_network_base import AsymmetricRMSELoss, BaseModel
from slrp_ev_data.feature_engineering import one_hot_encoding
from slrp_ev_data.window_generator import WindowGenerator
from torch.utils.data import DataLoader, Dataset

# PyTorch TensorBoard support
from tqdm import tqdm

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # type: ignore


# Suppress all warnings
# warnings.filterwarnings("ignore")


class FFNN(BaseModel):
    def __init__(
        self,
        x_dim: int = 16,
        lookahead: int = 16,
        alpha: int = 2,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        activation=nn.ReLU(),
        initial_learning_rate: float = 0.01,
        lr_threshold: float = 1e-5,
        scheduler_patience: int = 5,
        epochs: int = 100,
        number_of_initial_models: int = 5,
        time_mode: Literal["window", "cyclical"] = "cyclical",
        batch_size: int = 64,
    ):
        """TODO"""
        # Initialize the BaseModel with relevant parameters
        super().__init__(
            initial_learning_rate=initial_learning_rate,
            scheduler_patience=scheduler_patience,
            epochs=epochs,
            model_str_name="basic_ffnn",
            lr_threshold=lr_threshold,
        )

        # FFNN-specific parameters
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.output_size = lookahead
        self.num_hidden_layers = num_hidden_layers
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

    def _determine_input_size(self) -> int:
        """Determines the input size of the model based on the time_mode."""
        if self.time_mode == "window":
            return self.x_dim + 6 + 1  # 6 for time one-hot encoding, 1 for workday
        elif self.time_mode == "cyclical":
            return self.x_dim + 1 + 4  # 1 for workday, 4 for sin/cos encoding
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

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> None:
        """Train and find the best model from a set of initial models."""
        X_train, y_train = self.get_X_y(train, overlapping_windows=True)  # type: ignore
        train_loader = self.get_dataloader(
            X_train, y_train, batch_size=self.batch_size, shuffle=True
        )

        X_val, y_val = self.get_X_y(val, overlapping_windows=False)  # type: ignore
        val_loader = self.get_dataloader(
            X_val, y_val, batch_size=self.batch_size, shuffle=False
        )

        self.add_model_to_board(train_loader)

        self.best_vloss = np.inf
        for i in (pbar := tqdm(range(self.number_of_initial_models))):
            pbar.set_description_str(
                f"Training Initial Model {i + 1}/{self.number_of_initial_models}"
            )
            self.initialize_model()
            self.initialize_optimizer_scheduler()
            self.fit_one_model(
                train_loader, val_loader, epochs=3, best_vloss=self.best_vloss
            )

        # Resume training the best model
        self.load_checkpoint()
        self.fit_one_model(
            train_loader, val_loader, epochs=self.epochs, best_vloss=self.best_vloss
        )

    def predict(self, test: pd.DataFrame):
        """Predict function to return error metrics and predictions."""
        X_test, y_test, y_dates = self.get_X_y(  # type: ignore
            test, return_y_date=True, overlapping_windows=False
        )
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        self.load_checkpoint()  # Load the best model
        self.model.eval()

        with torch.no_grad():
            y_pred_test = self.model(X_test_tensor).detach().numpy().squeeze()
            y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32)

            # Calculate RMSE and weighted RMSE
            rmse = np.sqrt(self.criterion(y_test_tensor, y_pred_test_tensor).item())
            weighted_criterion = AsymmetricRMSELoss(alpha=self.alpha)
            wrmse = np.sqrt(
                weighted_criterion(y_test_tensor, y_pred_test_tensor).item()
            )

        # Flatten the lists to 1D
        y_pred_test_flat = y_pred_test.flatten()
        forecast_dates = y_dates.to_numpy().flatten()
        return rmse, wrmse, y_pred_test_flat, forecast_dates

    def get_X_y(
        self,
        df: pd.DataFrame,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame]
        | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ):
        W = WindowGenerator(
            input_width=self.x_dim,
            label_width=self.lookahead,
            shift=self.lookahead,
            train_df=df,
            label_columns=["power", "date"],
            overlapping_windows=overlapping_windows,
        )
        cols_keep_last_value = ["workday"]
        if self.time_mode == "cyclical":
            cols_keep_last_value += ["Day sin", "Day cos", "Year sin", "Year cos"]
        elif self.time_mode == "window":
            cols_keep_last_value += ["time_window"]

        flat_inputs, flat_labels = W.flatten_dataset(
            W.train,
            cols_to_flatten=["power"],
            cols_keep_last_value=cols_keep_last_value,
            label_cols_to_flatten=["power"],
        )
        if self.time_mode == "window":
            flat_inputs = one_hot_encoding(flat_inputs, ["time_window"])
        print(f"Input shape: {flat_inputs.shape}, label shape: {flat_labels.shape}")

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                W.train, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            return flat_inputs, flat_labels, y_dates
        else:
            return flat_inputs, flat_labels

    def get_dataloader(
        self,
        flat_x: pd.DataFrame,
        flat_y: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Given the output from self.get_X_y, returns a DataLoader object, which makes it easier to train
        data in batches

        Args:
            flat_x (pd.DataFrame): x output from self.get_X_y
            flat_y (pd.DataFrame): y output from self.get_X_y
            batch_size (int): _description_
            shuffle (bool, optional): shuffling is advisable for the training data, but not for the test/val data.
                Defaults to True.

        Returns:
            DataLoader: _description_
        """
        dataset = TensorDataset(flat_x, flat_y)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


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
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
