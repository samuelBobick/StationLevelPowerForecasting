from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
from slrp_ev_data.window_generator import TFToTorchDataset, WindowGenerator
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# Suppress all warnings
# warnings.filterwarnings("ignore")


class OldLSTM:

    def __init__(
        self,
        x_dim: int = 16,
        lookahead: int = 16,
        alpha: int = 2,
        hidden_size: int = 64,
        output_size: int = 16,
        num_lstm_layers: int = 1,
        activation=nn.ReLU(),
        initial_learning_rate: float = 0.01,
        scheduler_patience: int = 3,
        epochs: int = 100,
        time_mode: Literal["window", "cyclical"] = "cyclical",
        batch_size: int = 64,
    ):
        """
        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (float, optional): . Defaults to 2.

        Neural Net Parameters:
            hidden_size (int, optional): Defaults to 64.
            output_size (int, optional): Defaults to 16. TODO: Should be equal to lookahead (for now).
            num_hidden_layers (int, optional): Number of LSTM modules that are stacked. Defaults to 2.
            activation (_type_, optional): Activation function from pytorch. Defaults to nn.ReLU().
            initial_learning_rate (float, optional): Defaults to 0.01.
            epochs (int, optional): Defaults to 100.
            batc_size (int, optional): Defaults to 64.
        """
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_path = Path(__file__).parent / "model" / "lstm.pt"
        self.model_path.parent.mkdir(exist_ok=True)
        self.tensorboard_path = Path(__file__).parent / "runs"

        self.time_mode = time_mode
        self.scheduler_patience = scheduler_patience
        if time_mode == "window":
            input_size = (
                1 + 6 + 1
            )  # 1 for power +  6 for one-hot encoding of time of day, 1 for workday
        elif time_mode == "cyclical":
            input_size = (
                1 + 1 + 4
            )  # 1 for power + 1 for workday, 4 for sin/cos of time of day

        self.model_inputs = (
            output_size,
            input_size,
            hidden_size,
            num_lstm_layers,
            activation,
        )
        self.initial_learning_rate = initial_learning_rate

        self.criterion = nn.MSELoss()

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        number_of_initial_models: int = 5,
    ) -> None:
        """Find best model (out of number_of_initial_models) and train it on the entire dataset"""

        train_loader = self.get_dataloader(
            train, shuffle=True, overlapping_windows=True
        )

        if val is not None:
            val_loader = self.get_dataloader(val, shuffle=False)

        self.add_model_to_board(train_loader)

        self.best_vloss = np.inf
        # Train number_of_initial_models models and save the best one
        for i in (pbar := tqdm(range(number_of_initial_models))):
            pbar.set_description_str(
                f"Training Initial Model {i + 1}/{number_of_initial_models}"
            )
            # Initialize model
            self.model = LSTM_model(*self.model_inputs)
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.initial_learning_rate,
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", patience=self.scheduler_patience
            )
            self.fit_one_model(
                train_loader, val_loader, epochs=3, best_vloss=self.best_vloss
            )

        # At this point, we have started training number_of_initial_models and we saved the best one
        # We can load the checkpoint of the best one and resume training
        # Initialize model
        self.model = LSTM_model(*self.model_inputs)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.initial_learning_rate,
        )
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.scheduler_patience
        )
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        current_model_epoch = checkpoint["epoch"]
        self.fit_one_model(
            train_loader,
            val_loader,
            epochs=self.epochs - current_model_epoch,
            writer=self.best_model_writer,
            start_epoch=current_model_epoch + 1,
            best_vloss=self.best_vloss,
        )

    def fit_one_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        best_vloss: float = np.inf,
        epochs: int | None = None,
        writer: SummaryWriter | None = None,
        start_epoch: int = 0,
    ) -> None:
        """Train self.model using self.optimizer (adn self.scheduler)"""
        # Create tensorboard writer
        # Default log_dir argument is "runs" - but it's good to be specific
        if writer is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(
                self.tensorboard_path / "lstm_{}".format(timestamp), max_queue=3
            )

        if epochs is None:
            epochs = self.epochs

        for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs"):
            avg_loss = self._train_epoch(train_loader, writer, epoch)

            avg_vloss = self._get_val_loss(val_loader)

            # Log the running loss averaged per batch
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Validation": avg_vloss},  # "Training": avg_loss,
                (epoch + 1) * len(train_loader),
            )
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                self.best_vloss = best_vloss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                    self.model_path,
                )
                # save model writer to be able to resume training later
                self.best_model_writer = writer

            # Log loss
            if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
                tqdm.write(
                    f"Epoch [{epoch + 1}/{epochs}], Training loss: {avg_loss:_.4f}, Validation loss:{avg_vloss:_.4f}"
                )
            # Change learning rate if validation loss plateaus
            current_lr = self.scheduler.get_last_lr()
            self.scheduler.step(avg_vloss)
            next_lr = self.scheduler.get_last_lr()
            if next_lr[0] < 1e-5:
                tqdm.write(
                    f"Epoch [{epoch + 1}/{epochs}], Learning rate is too small,"
                    + "stopping training"
                )
                break
            if next_lr != current_lr:
                tqdm.write(
                    f"Epoch [{epoch + 1}/{epochs}], Learning rate changed from {current_lr} to {next_lr}"
                )

        print(f"Training complete! Lowest validation loss is: {best_vloss}")

    def _train_epoch(self, train_loader, writer, epoch) -> float:
        running_loss = 0.0
        self.model.train(
            True
        )  # Switching to training mode, eg. turning on regularisation

        for batch_number, batch in enumerate(train_loader):
            x_batch, y_batch = batch

            # reshape y to have same shape as output (remove the 1 dimension at the end)
            y_batch = y_batch.squeeze()

            self.optimizer.zero_grad()
            # Forward pass
            y_pred_batch = self.model(x_batch)

            loss = self.criterion(y_pred_batch.squeeze(), y_batch)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": loss.item()},
                (epoch) * len(train_loader) + (batch_number + 1),
            )

        return running_loss / len(train_loader)

    def _get_val_loss(self, val_loader) -> float:
        # Check against the validation set
        running_vloss = 0.0

        # In evaluation mode some model specific operations can be omitted eg. dropout layer
        self.model.train(False)
        with torch.no_grad():
            for val_batch_number, val_batch in enumerate(val_loader):
                val_x_batch, val_y_batch = val_batch
                # reshape y to have same shape as output (remove the 1 dimension at the end)
                val_y_batch = val_y_batch.squeeze()
                val_y_pred_batch = self.model(val_x_batch)
                vloss = self.criterion(val_y_pred_batch.squeeze(), val_y_batch)
                running_vloss += vloss.item()

        return running_vloss / len(val_loader)

    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        dataset, y_dates = self.get_dataset(
            test, return_y_date=True, overlapping_windows=False
        )
        X_test_tensor, y_test_tensor = dataset.get_full_data()
        y_test_tensor = y_test_tensor.squeeze()

        self.model = LSTM_model(*self.model_inputs)
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        y_pred_test = self.model(X_test_tensor).detach().numpy().squeeze()
        y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32)
        rmse = np.sqrt(
            self.criterion.forward(
                y_test_tensor,
                y_pred_test_tensor,
            ).item()
        )

        weighted_criterion = AsymmetricRMSELoss(alpha=2)
        wrmse = np.sqrt(
            weighted_criterion.forward(
                y_pred_test_tensor,
                y_test_tensor,
            ).item()
        )

        # Flatten the lists to 1D
        y_pred_test_flat = y_pred_test.flatten()
        forecast_dates = y_dates.to_numpy().flatten()
        return rmse, wrmse, y_pred_test_flat, forecast_dates

    def get_dataset(
        self,
        df,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, pd.DataFrame]:
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

        # TODO review the following for time_mode == "window" bc flattening is no longer necessary
        # flat_inputs, flat_labels = W.flatten_dataset(
        #     W.train,
        #     cols_to_flatten=["power"],
        #     label_cols_to_flatten=["power"],
        # )
        # if self.time_mode == "window":
        #     flat_inputs = one_hot_encoding(flat_inputs, ["time_window"])
        # print(f"Input shape: {flat_inputs.shape}, label shape: {flat_labels.shape}")
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
        df,
        shuffle: bool = False,
        overlapping_windows: bool = False,
    ) -> DataLoader:
        """Given the output from self.get_X_y, returns a DataLoader object, which makes it easier to train
        data in batches

        Args:
            flat_x (pd.DataFrame): x output from self.get_X_y
            flat_y (pd.DataFrame): y output from self.get_X_y
            shuffle (bool, optional): shuffling is advisable for the training data, but not for the test/val data.
                Defaults to True.
            overlapping_windows (bool, optional): _description_. Defaults to False.

        Returns:
            DataLoader: Data put into batches, ready for training
        """
        dataset = self.get_dataset(df, overlapping_windows=overlapping_windows)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)  # type: ignore
        return dataloader

    def add_model_to_board(self, train_loader: DataLoader) -> None:
        writer = SummaryWriter(self.tensorboard_path / "lstm_model_schema")
        # get some random training images
        dataiter = iter(train_loader)
        inputs, labels = next(dataiter)

        model = LSTM_model(*self.model_inputs)
        writer.add_graph(model, inputs)
        # Count the number of trainable parameters and add it to TensorBoard as a scalar
        total_params = count_parameters(model)
        writer.add_scalar("Model/Total_Parameters", total_params)

        writer.flush()
        writer.close()


def count_parameters(model):
    # Function to calculate the total number of parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LSTM_model(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, activation):
        super(LSTM_model, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # number of hidden units
        self.activation = activation

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, output_size)  # fully connected last layer

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
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state

        # hn is of size [num_layers, batch_size, hidden_size]
        hn = hn[-1]  # reshaping the data for Dense layer next
        out = self.activation(hn)
        out = self.fc_1(out)  # first Dense
        out = self.activation(out)
        out = self.fc(out)  # Final Output
        out = nn.functional.softplus(
            out
        )  # because we don't want the final values to be negative
        return out


class AsymmetricRMSELoss(nn.Module):
    def __init__(self, alpha):
        super(AsymmetricRMSELoss, self).__init__()
        self.multiplier = alpha

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        loss = torch.sqrt(
            torch.mean(
                torch.pow(self.multiplier, 1 - torch.sign(input - target)) * mse_loss
            )
        )
        return loss
