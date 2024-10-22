from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from slrp_ev_ts_forecasting.asymmetric_loss import AsymmetricRMSELoss
from slrp_ev_ts_forecasting.base import Base
from slrp_ev_ts_forecasting.compute_losses import Losses, compute_torch_losses
from slrp_ev_ts_forecasting.default_parameters import (
    DEVICE,
    TypeErrorMetric,
    TypeOptimizeLags,
)

# PyTorch TensorBoard support
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # type: ignore


# Parent class for common functionality
class TorchBaseModel(Base):
    def __init__(
        self,
        epochs: int,
        number_of_initial_models: int,
        batch_size: int,
        model_str_name: str,
        alpha: int,
        initial_learning_rate: float,
        lr_threshold: float,
        scheduler_patience: int,
        error_metric: TypeErrorMetric,
        x_dim: int,
        lookahead: int,
        warmup_base_learning_rate: float = 1e-6,
        optimize_lags: TypeOptimizeLags = None,
    ):
        super().__init__(x_dim=x_dim, lookahead=lookahead, optimize_lags=optimize_lags)
        # General NN parameters
        self.epochs = epochs
        self.number_of_initial_models = number_of_initial_models
        if number_of_initial_models == 1:
            self.epochs_initial_models = epochs
        else:
            self.epochs_initial_models = 3
            assert (
                self.epochs >= self.epochs_initial_models
            ), f"Epochs must be greater than {self.epochs_initial_models}, the number of epochs for initial models."
        self.batch_size = batch_size

        # Weighted loss parameters
        self.alpha = alpha
        if error_metric == "mse":
            self.criterion = nn.MSELoss()
        elif error_metric == "wmse":
            self.criterion = AsymmetricRMSELoss(alpha)
        else:
            raise ValueError(
                f"Error metric of type {error_metric} is not defined. Please refer "
                "to TypeErrorMetric for supported error metrics."
            )
        # Path parameters
        self.model_path = Path(__file__).parent / "model" / f"{model_str_name}.pt"
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.tensorboard_path = Path(__file__).parent / "runs"

        # Lr scheduler parameters
        # self.scheduler_patience = scheduler_patience
        self.initial_learning_rate = initial_learning_rate
        self.lr_threshold = lr_threshold

        # Warmup lr parameters
        self.warmup_base_learning_rate = warmup_base_learning_rate
        self.warmup_steps = 1000

        # These will be implemented in child classes
        self.model: Optional[nn.Module] = None

        # Parameters for optimize lags for regression models (e.g. FFNN)
        self.optimize_lags = optimize_lags

    def initialize_optimizer_scheduler(self):
        """Initialize the optimizer and learning rate scheduler.
        Raises NotImplementedError if the model is not set in the child class.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model must be implemented in the subclass before initializing the optimizer and scheduler."
            )
        # TODO: try eps=1e-4 instead of the default 1e-8 (as shown in a paper)
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.initial_learning_rate, eps=1e-8
        )

        # Define the warmup scheduler
        def _lr_lambda(step_number):
            if step_number < self.warmup_steps:
                # Interpolate linearly between warmup_base_learning_rate and initial_learning_rate
                return (
                    self.warmup_base_learning_rate
                    + (self.initial_learning_rate - self.warmup_base_learning_rate)
                    * (step_number / self.warmup_steps)
                ) / self.initial_learning_rate
            return 1.0  # After warmup, use the normal learning rate

        self.warmup_scheduler = lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        # self.scheduler = lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, "min", patience=self.scheduler_patience
        # )

        # Cosine annealing scheduler without restarts
        # we subtract because this scheduler will only start after the warmup phase
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * self.number_of_steps_per_epoch - self.warmup_steps,
            eta_min=1e-7,
        )

    def save_checkpoint(self, epoch: int, best_vloss: float) -> None:
        """Saves the model, optimizer, and scheduler to a checkpoint."""
        if self.model is None:
            raise NotImplementedError(
                "Model, optimizer, and scheduler must be implemented before saving the checkpoint."
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_vloss": best_vloss,
            },
            self.model_path,
        )

    def load_checkpoint(self) -> int:
        """Loads the model, optimizer, and scheduler from a checkpoint."""
        if self.model is None:
            raise NotImplementedError(
                "Model, optimizer, and scheduler must be implemented before loading the checkpoint."
            )

        checkpoint = torch.load(self.model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_vloss = checkpoint["best_vloss"]
        return checkpoint["epoch"]

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> None:
        """Find best model (out of number_of_initial_models) and train it on the entire dataset."""
        if self.optimize_lags:
            self.pacf_top_values = self.get_top_pacf_values(train)

        train_loader = self.get_dataloader(
            train, shuffle=True, overlapping_windows=True
        )
        self.update_seen_data(train)
        self.number_of_steps_per_epoch = len(train_loader)
        val_loader = self.get_dataloader(val, shuffle=False, overlapping_windows=False)
        self.update_seen_data(val)

        self.initialize_model()
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
                train_loader,
                val_loader,
                epochs=self.epochs_initial_models,
                best_vloss=self.best_vloss,
            )

        if self.number_of_initial_models > 1:
            # At this point, we have started training number_of_initial_models and we saved the best one
            # We can load the checkpoint of the best one and resume training
            current_model_epoch = self.load_checkpoint()
            self.fit_one_model(
                train_loader,
                val_loader,
                start_epoch=current_model_epoch + 1,
                writer=self.best_model_writer,
                best_vloss=self.best_vloss,
            )

    def fit_one_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        best_vloss: float = np.inf,
        epochs: Optional[int] = None,
        writer: Optional[SummaryWriter] = None,
        start_epoch: int = 0,
    ) -> None:
        if writer is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(
                self.tensorboard_path / f"{self.model_str_name}_{timestamp}",
                max_queue=5,
            )

        if epochs is None:
            epochs = self.epochs

        if self.model is None:
            raise NotImplementedError(
                "Model, optimizer, and scheduler must be implemented before fitting."
            )

        for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs"):
            avg_loss, avg_vloss = self._train_epoch(
                train_loader, val_loader, writer, epoch
            )

            current_lr = self._get_current_lr()

            # log loss
            tqdm.write(
                f"Epoch [{epoch + 1}/{epochs}], Current lr: {current_lr:.2E}, Training loss: {avg_loss:_.4f}, Validation loss:{avg_vloss:_.4f}"
            )

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                self.best_vloss = best_vloss
                self.save_checkpoint(epoch, best_vloss)
                self.best_model_writer = writer

            # early stopping criteria
            # next_lr = self.scheduler.get_last_lr()[0]

            # if next_lr < 1e-5:
            #     tqdm.write(
            #         f"Epoch [{epoch + 1}/{epochs}], Learning rate is too small, stopping training"
            #     )
            #     break

        print(f"Training complete! Lowest validation loss is: {best_vloss}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        writer: SummaryWriter,
        epoch: int,
    ) -> tuple[float, float]:
        if self.model is None or self.optimizer is None:
            raise NotImplementedError(
                "Model and optimizer must be implemented before training."
            )

        running_loss = 0.0
        self.model.train()

        for batch_number, batch in enumerate(train_loader):
            x_batch, y_batch = batch
            step_number = epoch * len(train_loader) + (batch_number + 1)

            # reshape y to have same shape as output (remove the 1 dimension at the end)
            y_batch = y_batch.squeeze()

            self.optimizer.zero_grad()

            # Forward pass
            y_pred_batch = self.model(x_batch)
            loss = self.criterion(
                y_pred_batch.squeeze()[
                    :, self.first_prediction_index :
                ],  # [:, self.first_prediction_index :]
                y_batch[:, self.first_prediction_index :],  #
            )

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if (batch_number + 1) % 50 == 0 or (
                batch_number == 0
            ):  # Log every 50 batches
                avg_vloss = self._get_val_loss(val_loader)

                writer.add_scalar(
                    "Training Loss",
                    loss.item(),
                    step_number,
                )
                writer.add_scalar(
                    "Validation Loss",
                    avg_vloss,
                    step_number,
                )

                # Log the current learning rate
                current_lr = self._get_current_lr()
                writer.add_scalar("Learning Rate", current_lr, step_number)

            if step_number < self.warmup_steps:
                self.warmup_scheduler.step()  # Linear warmup
            else:
                # Apply normal scheduler if we are after the warmup phase
                self.scheduler.step()  # CosineAnnealingLR adjusts per epoch
                # if avg_vloss:
                #     self.scheduler.step(avg_vloss) # for Plateau LR

        return running_loss / len(train_loader), avg_vloss

    def _get_val_loss(self, val_loader) -> float:
        if self.model is None:
            raise NotImplementedError("Model must be implemented before validating.")

        running_vloss = 0.0
        self.model.eval()
        with torch.no_grad():
            for val_batch_number, val_batch in enumerate(val_loader):
                val_x_batch, val_y_batch = val_batch
                # reshape y to have same shape as output (remove the 1 dimension at the end)
                val_y_batch = val_y_batch.squeeze()
                val_y_pred_batch = self.model(val_x_batch)
                vloss = self.criterion(
                    val_y_pred_batch.squeeze()[:, self.first_prediction_index :],
                    val_y_batch[:, self.first_prediction_index :],
                )
                running_vloss += vloss.item()
        self.model.train()

        return running_vloss / len(val_loader)

    def _get_current_lr(self) -> float:
        # all layers have the same lr so we can just return the lr of the first layer
        return self.optimizer.param_groups[0]["lr"]  # type: ignore

    def predict(self, test: pd.DataFrame) -> tuple[Losses, np.ndarray, np.ndarray]:
        """Given a pandas DataFrame test, returns error metrics and list of predictions."""
        dataset, y_dates = self.get_dataset(
            test, return_y_date=True, overlapping_windows=False
        )
        X_test_tensor, y_test_tensor = dataset.get_full_data()  # type: ignore
        y_test_tensor = y_test_tensor.squeeze()[:, self.first_prediction_index :]

        # Load model from the checkpoint
        self.load_checkpoint()
        print(f"Best validation loss of model retrieved: {self.best_vloss}")

        self.model.eval()
        y_pred_test = self.model(X_test_tensor).detach().cpu().numpy().squeeze()
        y_pred_test_tensor = torch.tensor(
            y_pred_test, dtype=torch.float32, device=DEVICE
        )
        y_pred_test_tensor = y_pred_test_tensor[:, self.first_prediction_index :]

        losses = compute_torch_losses(
            y_pred_test_tensor.flatten(), y_test_tensor.flatten(), self.alpha
        )

        # Flatten the lists to 1D
        y_pred_test_flat = y_pred_test_tensor.flatten().cpu().numpy()
        y_dates = y_dates.squeeze()[:, self.first_prediction_index :]
        forecast_dates = (
            y_dates[:, self.first_prediction_index :].flatten().cpu().numpy()
        )

        return losses, y_pred_test_flat, forecast_dates

    def add_model_to_board(self, train_loader: DataLoader) -> None:
        if self.model is None:
            raise NotImplementedError(
                "Model must be implemented before adding it to TensorBoard."
            )

        writer = SummaryWriter(self.tensorboard_path / f"{self.model_str_name}_schema")
        dataiter = iter(train_loader)
        inputs, labels = next(dataiter)

        print(
            "Model size",
            f"    Size of train set: {len(train_loader)} batches of size {self.batch_size}",
            f"    therefore, we have {len(train_loader)} steps at each of the {self.epochs} epochs.",
            f"    Train input shape: {inputs.shape}",
            f"    Train label shape: {labels.shape}",
            sep="\n",
        )

        writer.add_graph(self.model, inputs)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total model parameters: {total_params}")
        writer.add_scalar("Model/Total_Parameters", total_params)
        writer.flush()
        writer.close()

    def get_dataloader(
        self,
        df: pd.DataFrame,
        shuffle: bool = False,
        overlapping_windows: bool = False,
    ) -> DataLoader:
        """Given the dataset, returns a DataLoader object."""
        dataset: Dataset = self.get_dataset(
            df, overlapping_windows=overlapping_windows
        )  # type: ignore
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    @property
    def seen_data(self):
        seen_data = getattr(self, "_seen_data", None)
        if seen_data is None:
            self._seen_data = pd.DataFrame(columns=["date"])
        return self._seen_data

    def update_seen_data(self, data: pd.DataFrame) -> None:
        # Concatenate the DataFrames
        concatenated_data = pd.concat([self.seen_data, data], ignore_index=True)

        # Identify duplicated dates
        duplicated_dates = concatenated_data[
            concatenated_data.duplicated(subset="date", keep=False)
        ]
        if not duplicated_dates.empty:
            print(
                f"Warning: {len(duplicated_dates)} duplicated dates found in the data. Dropping duplicates."
            )
            concatenated_data = concatenated_data.drop_duplicates(subset="date")

        # Assign the combined DataFrame to self._seen_data
        self._seen_data = concatenated_data

    def pad_with_seen_data(
        self, new_data_to_pad: pd.DataFrame, number_of_timesteps_to_pad: int
    ) -> pd.DataFrame:
        """Add at the beginning of the "new_data_to_pad" DataFrame the "number_of_timesteps_to_pad" that precede the given data.
        If the data is not available or some timesteps are missing, no padding is done.
        """
        first_date_of_data_to_pad = pd.to_datetime(
            new_data_to_pad.iloc[0]["date"], unit="s"
        )
        # Build padding index
        padding_index = pd.date_range(
            start=first_date_of_data_to_pad
            - pd.Timedelta(minutes=15) * number_of_timesteps_to_pad,
            periods=number_of_timesteps_to_pad,
            freq="15min",
        )

        seen_data = self.seen_data.copy()
        seen_data["date"] = pd.to_datetime(seen_data["date"], unit="s")
        seen_data["date"] = seen_data["date"].dt.round("5min")
        seen_data = seen_data.set_index("date")

        # Check if the padding index is in the model data
        if not padding_index.isin(seen_data.index).all():
            if not seen_data.empty:
                print(
                    "Warning: Some padding indexes are missing in the model data. Padding not done."
                )
            return new_data_to_pad

        # Get the padding data
        padding_data = seen_data.loc[padding_index]
        padding_data = padding_data.reset_index().rename(columns={"index": "date"})
        padding_data["date"] = padding_data["date"].astype("int64") // 10**9

        # Concatenate the padding data with the data to pad
        new_data_to_pad = pd.concat([padding_data, new_data_to_pad], ignore_index=True)

        return new_data_to_pad

    def get_dataset(
        self,
        df: pd.DataFrame,
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> Dataset | tuple[Dataset, torch.Tensor]:
        raise NotImplementedError("This method must be implemented in the child class.")

    def initialize_model(self) -> None:
        raise NotImplementedError("This method must be implemented in the child class.")

    @property
    def model_str_name(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    @property
    def first_prediction_index(self) -> int:
        """This property exists to allow for model outputs that contain some of
        the input data.
        For instance:
            - The input looks like this: [0, 1, 2, 3, 4, 5]
            - You want to predict the next 3 values: [6, 7, 8]
            - But the output of the model is of the same length as the input so
            it looks like [3, 4, 5, 6, 7, 8]
            - In this case, the first_prediction_index should be 3
        in the case the model output is of the desired output length (3 in this case)
        the first_prediction_index should be 0. This is the default value.

        This property can be redefined in the child class in case the model output length
        is different from the desired output length.
        """
        # This property can be redefined in the child class
        return 0
