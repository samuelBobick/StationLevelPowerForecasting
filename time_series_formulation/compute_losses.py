from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from asymmetric_loss import AsymmetricRMSELoss, asymmetric_rmse
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Losses(TypedDict):
    rmse: float
    wrmse: float
    mae: float


def compute_losses(
    y_pred: np.ndarray | list, y_true: np.ndarray | list, alpha: float
) -> Losses:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wrmse = asymmetric_rmse(alpha, y_pred, y_true)
    mae = mean_absolute_error(y_true, y_pred)
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae)  # type: ignore


def compute_torch_losses(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float
) -> Losses:
    rmse = torch.sqrt(nn.functional.mse_loss(y_pred, y_true, reduction="mean")).item()
    wrmse = AsymmetricRMSELoss(alpha)(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae)


def get_real_scale_losses(losses: Losses, normalize_parameters) -> Losses:
    train_min, train_max = normalize_parameters
    losses["rmse"] = (
        losses["rmse"] * (train_max["power"] - train_min["power"]) + train_min["power"]
    )
    losses["wrmse"] = (
        losses["wrmse"] * (train_max["power"] - train_min["power"]) + train_min["power"]
    )
    losses["mae"] = (
        losses["mae"] * (train_max["power"] - train_min["power"]) + train_min["power"]
    )
    return losses
