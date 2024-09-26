from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


# Parent class for common functionality
class BaseModel:
    def __init__(
        self,
        epochs: int,
        model_str_name: str,
        initial_learning_rate: float,
        lr_threshold: float,
        scheduler_patience: int,
        warmup_base_learning_rate: float = 1e-6,
    ):

        self.epochs = epochs
        self.scheduler_patience = scheduler_patience
        self.initial_learning_rate = initial_learning_rate
        self.lr_threshold = lr_threshold

        self.criterion = nn.MSELoss()
        self.best_vloss = np.inf

        self.model_str_name = model_str_name
        self.model_path = Path(__file__).parent / "model" / f"b{model_str_name}.pt"
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.tensorboard_path = Path(__file__).parent / "runs"

        # warmup lr parameters
        self.warmup_base_learning_rate = warmup_base_learning_rate
        self.warmup_steps = 1000

        # These will be implemented in child classes
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None

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
        # we divide self.epochs by 1.1 because with the warmup, the
        # scheduler will be called slightly less than the number of epochs
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(self.epochs // 1.1), eta_min=1e-7
        )

    def save_checkpoint(self, epoch: int, best_vloss: float) -> None:
        """Saves the model, optimizer, and scheduler to a checkpoint."""
        if self.model is None or self.optimizer is None or self.scheduler is None:
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
        if self.model is None or self.optimizer is None or self.scheduler is None:
            raise NotImplementedError(
                "Model, optimizer, and scheduler must be implemented before loading the checkpoint."
            )

        checkpoint = torch.load(self.model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_vloss = checkpoint["best_vloss"]
        return checkpoint["epoch"]

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

        if self.model is None or self.optimizer is None or self.scheduler is None:
            raise NotImplementedError(
                "Model, optimizer, and scheduler must be implemented before fitting."
            )

        number_of_steps_per_epoch = len(train_loader)

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

            # Apply scheduler if we are after the warmup phase
            if epoch > self.warmup_steps / number_of_steps_per_epoch:
                self.scheduler.step()  # CosineAnnealingLR adjusts per epoch
                # self.scheduler.step(avg_vloss) for Plateau LR
                next_lr = self.scheduler.get_last_lr()[0]

                if next_lr < 1e-5:
                    tqdm.write(
                        f"Epoch [{epoch + 1}/{epochs}], Learning rate is too small, stopping training"
                    )
                    break
                if next_lr != current_lr:
                    tqdm.write(
                        f"Epoch [{epoch + 1}/{epochs}], Learning rate changed from {current_lr} to {next_lr}"
                    )

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
            loss = self.criterion(y_pred_batch.squeeze(), y_batch)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if step_number < self.warmup_steps:
                self.warmup_scheduler.step(step_number)

            if (batch_number + 1) % 50 == 0 or (
                batch_number == 0
            ):  # Log every 50 batches
                avg_vloss = self._get_val_loss(val_loader)

                # Log the current learning rate
                current_lr = self._get_current_lr()

                writer.flush()
                writer.add_scalars(
                    "Training vs. Validation Loss",
                    {
                        "Training": loss.item(),
                        "Validation": avg_vloss,
                        "LR": current_lr,
                    },
                    step_number,
                )

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
                vloss = self.criterion(val_y_pred_batch.squeeze(), val_y_batch)
                running_vloss += vloss.item()
        self.model.train()

        return running_vloss / len(val_loader)

    def _get_current_lr(self) -> float:
        # all layers have the same lr so we can just return the lr of the first layer
        return self.optimizer.param_groups[0]["lr"]  # type: ignore

    def add_model_to_board(self, train_loader: DataLoader) -> None:
        if self.model is None:
            raise NotImplementedError(
                "Model must be implemented before adding it to TensorBoard."
            )

        writer = SummaryWriter(self.tensorboard_path / f"{self.model_str_name}_schema")
        dataiter = iter(train_loader)
        inputs, _ = next(dataiter)

        writer.add_graph(self.model, inputs)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total model parameters: {total_params}")
        writer.add_scalar("Model/Total_Parameters", total_params)
        writer.flush()
        writer.close()


class AsymmetricRMSELoss(nn.Module):
    def __init__(self, alpha):
        super(AsymmetricRMSELoss, self).__init__()
        self.multiplier = alpha**2

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        residual = input - target
        mask = residual <= 0  # mask for underpredictions
        loss = torch.sqrt(
            torch.mean((1 + (self.multiplier - 1) * mask.float()) * mse_loss)
        )
        return loss
