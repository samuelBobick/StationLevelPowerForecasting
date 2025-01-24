from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from slrp_ev_data.window_generator import TFToTorchDataset
from slrp_ev_ts_forecasting.asymmetric_loss import AsymmetricRMSELoss
from slrp_ev_ts_forecasting.default_parameters import (
    SAVED_MODELS_PATH,
    TypeErrorMetric,
    TypeOptimizeLags,
    TypeScalingMode,
)
from slrp_ev_ts_forecasting.helper_session_forecasting import get_artificial_data
from slrp_ev_ts_forecasting.models.base import Base
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

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
        optimize_lags: TypeOptimizeLags,
        time_mode: Literal["window", "cyclical"],
        get_val_data_from_shuffled_train: bool,
        scaling_mode: TypeScalingMode,
        scaling_parameters: tuple | pd.DataFrame | None,
        session_based_mode: bool,
        peak_prediction: bool,
        add_number_of_sessions: bool,
        add_fraction_of_regular_sessions: bool,
        use_all_active_sessions: bool,
        number_of_artificial_datasets: int,
        random_start_time: bool,
        shuffle_power_profiles: bool,
        random_power_profile_shapes: bool,
        random_user_needs: bool,
        random_choices: bool,
        add_number_of_evses_available: bool,
        warmup_base_learning_rate: float = 1e-6,
        use_decoder: bool = True,
    ):
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
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
        self.use_decoder = use_decoder  # can only be False for TCN

        self.time_mode = time_mode

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
        self.model_path = (
            SAVED_MODELS_PATH / "pytorch_saved_models" / f"{model_str_name}.pt"
        )
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
        self.peak_prediction = peak_prediction

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
            train, data_type="train", shuffle=True, overlapping_windows=True
        )
        self.update_seen_data(train)
        self.number_of_steps_per_epoch = len(train_loader)

        val_loader = self.get_dataloader(
            val, data_type="val", shuffle=False, overlapping_windows=False
        )
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
            )

    def fit_one_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
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

            # early stopping criteria
            # next_lr = self.scheduler.get_last_lr()[0]

            # if next_lr < 1e-5:
            #     tqdm.write(
            #         f"Epoch [{epoch + 1}/{epochs}], Learning rate is too small, stopping training"
            #     )
            #     break

        print(f"Training complete! Lowest validation loss is: {self.best_vloss}")

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

            self.optimizer.zero_grad()

            # Forward pass
            y_pred_batch = self.model(x_batch)

            # reshape y to have same shape as output (remove the 1 dimension at the end)
            if y_batch.dim() == 3:
                y_batch = y_batch.squeeze(-1)
            if y_pred_batch.dim() == 3:
                y_pred_batch = y_pred_batch.squeeze(-1)

            loss = self.criterion(
                y_pred_batch[
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

                if avg_vloss < self.best_vloss:
                    self.best_vloss = avg_vloss
                    self.save_checkpoint(epoch, self.best_vloss)
                    self.best_model_writer = writer

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
                val_y_pred_batch = self.model(val_x_batch)

                # reshape y to have same shape as output (remove the 1 dimension at the end)
                if val_y_batch.dim() == 3:
                    val_y_batch = val_y_batch.squeeze(-1)
                if val_y_pred_batch.dim() == 3:
                    val_y_pred_batch = val_y_pred_batch.squeeze(-1)

                vloss = self.criterion(
                    val_y_pred_batch[:, self.first_prediction_index :],
                    val_y_batch[:, self.first_prediction_index :],
                )
                running_vloss += vloss.item()
        self.model.train()

        return running_vloss / len(val_loader)

    def _get_current_lr(self) -> float:
        # all layers have the same lr so we can just return the lr of the first layer
        return self.optimizer.param_groups[0]["lr"]  # type: ignore

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Given a pandas DataFrame test, returns error metrics and list of predictions."""
        dataset, y_dates = self.get_dataset(
            test, data_type="test", return_y_date=True, overlapping_windows=False
        )  # type: ignore
        dataset: TFToTorchDataset
        y_dates: pd.DataFrame

        X_test_tensor, y_test_tensor = dataset.get_full_data()
        if y_test_tensor.dim() == 3:
            y_test_tensor = y_test_tensor.squeeze(-1)
        y_test_tensor = y_test_tensor[:, self.first_prediction_index :]

        # Load model from the checkpoint
        self.load_checkpoint()
        print(f"Best validation loss of model retrieved: {self.best_vloss}")

        if self.model is None:
            raise NotImplementedError("Model must be implemented before predicting.")

        self.model.eval()
        y_pred_test_tensor = self.model(X_test_tensor)
        if y_pred_test_tensor.dim() == 3:
            y_pred_test_tensor = y_pred_test_tensor.squeeze(-1)
        y_pred_test_tensor = y_pred_test_tensor[:, self.first_prediction_index :]

        # losses = compute_torch_losses(
        #     y_pred_test_tensor.flatten(), y_test_tensor.flatten(), self.alpha
        # )

        forecasts = y_pred_test_tensor.detach().numpy()
        reals = y_test_tensor.cpu().numpy()
        if len(y_dates.shape) == 1:
            y_dates = y_dates.to_frame()  # type: ignore
        y_dates = y_dates.iloc[:, self.first_prediction_index :]

        df_predictions = self.prepare_df_predictions(forecasts, y_dates, reals)

        return df_predictions

    def add_model_to_board(self, train_loader: DataLoader) -> None:
        if self.model is None:
            raise NotImplementedError(
                "Model must be implemented before adding it to TensorBoard."
            )

        writer = SummaryWriter(self.tensorboard_path / f"{self.model_str_name}_schema")
        data_iter = iter(train_loader)
        inputs, labels = next(data_iter)

        print(
            "Model size",
            f"    Size of train set: {len(train_loader)} batches of size {self.batch_size}"
            f" (there are around {(len(train_loader) * self.batch_size):,} training samples)",
            # actually we have slightly less samples because the last batch is smaller
            f"    therefore, we have {len(train_loader)} steps at each of the {self.epochs} epochs.",
            f"    Train input shape: {inputs.shape}",
            f"    Train label shape: {labels.shape}",
            sep="\n",
        )

        writer.add_graph(self.model, inputs)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Total model parameters: {total_params}. As a rule of thumbs, make sure you have 50 time less features, and 20 more samples"
        )
        writer.add_scalar("Model/Total_Parameters", total_params)
        writer.flush()
        writer.close()

    def get_dataloader(
        self,
        df: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        shuffle: bool = False,
        overlapping_windows: bool = False,
    ) -> DataLoader:
        """Given the dataset, returns a DataLoader object."""
        dataset: Dataset = self.get_dataset(df, data_type=data_type, overlapping_windows=overlapping_windows)  # type: ignore
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_dataset(
        self,
        df: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, pd.DataFrame]:
        """Generates the dataset and features based on the input DataFrame."""
        if df is not None:
            df = df.copy()
            df_padded = self.pad_with_seen_data(
                df, number_of_timesteps_to_pad=self.x_dim
            )
        else:
            df_padded = None

        samples = self.get_one_dataset(
            df_padded, data_type, return_y_date, overlapping_windows
        )
        if len(samples) == 2:
            return samples
        else:
            dataset = samples

        if data_type == "train":
            if return_y_date:
                raise ValueError(
                    "return_y_date is not yet supported for 'train' data. Please set it to False"
                )
            if df_padded is None:
                raise ValueError(
                    "df_padded should be provided to generate windows for train data type"
                )

            for i in range(self.number_of_artificial_datasets):
                artificial_df, _, _ = get_artificial_data(
                    train_data=df_padded,
                    random_start_time=self.random_start_time,
                    shuffle_power_profiles=self.shuffle_power_profiles,
                    random_power_profile_shapes=self.random_power_profile_shapes,
                    random_user_needs=self.random_user_needs,
                    random_choices=self.random_choices,
                    scaling_mode=self.scaling_mode,
                    lookahead=self.lookahead,
                )
                artificial_dataset: TFToTorchDataset = self.get_one_dataset(
                    artificial_df, data_type, return_y_date, overlapping_windows
                )  # type: ignore
                dataset = dataset + artificial_dataset
        return dataset  # type: ignore

    def get_one_dataset(
        self,
        df_padded: pd.DataFrame | None,
        data_type: Literal["train", "val", "test"],
        return_y_date: bool = False,
        overlapping_windows: bool = False,
    ) -> TFToTorchDataset | tuple[TFToTorchDataset, pd.DataFrame]:
        label_width = self.lookahead
        if not self.use_decoder:
            # To have an output of the same length as the input
            label_width = self.x_dim

        W, window_data = self.get_window_data(
            df_padded, self.x_dim, label_width, overlapping_windows, data_type
        )

        cols_to_keep_as_features = ["power"]
        if self.add_number_of_evses_available:
            cols_to_keep_as_features.append("number_of_evses_available")
        if self.time_mode == "cyclical":
            cols_to_keep_as_features += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ] + [col for col in self.list_workday_column_names if col != "workday_0"]
        elif self.time_mode == "window":
            cols_to_keep_as_features += ["time_window"] + self.list_workday_column_names

        dataset = W.convert_to_torch_dataset(
            window_data, cols_to_keep_as_features, cols_to_keep_as_labels=["power"]
        )

        if return_y_date:
            x_dates, y_dates = W.flatten_dataset(
                window_data, cols_to_flatten=["date"], label_cols_to_flatten=["date"]
            )
            return dataset, y_dates
        else:
            return dataset

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
