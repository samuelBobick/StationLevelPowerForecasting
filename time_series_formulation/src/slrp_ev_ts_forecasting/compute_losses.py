from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from asymmetric_loss import (
    AsymmetricRMSELoss,
    WeightedPeaksRMSELoss,
    asymmetric_rmse,
    weighted_peaks_rmse,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torcheval.metrics import R2Score


class Losses(TypedDict):
    rmse: float
    wrmse: float
    mae: float
    wprmse: float
    r2: float


def compute_losses(
    y_pred: np.ndarray | list, y_true: np.ndarray | list, alpha: float
) -> Losses:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wrmse = asymmetric_rmse(alpha, y_pred, y_true)
    mae = mean_absolute_error(y_true, y_pred)
    wprmse = weighted_peaks_rmse(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2)  # type: ignore


def compute_torch_losses(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float
) -> Losses:
    rmse = torch.sqrt(nn.functional.mse_loss(y_pred, y_true, reduction="mean")).item()
    wrmse = AsymmetricRMSELoss(alpha)(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    wprmse = WeightedPeaksRMSELoss()(y_pred, y_true).item()
    r2 = R2Score().update(y_pred, y_true).compute().item()
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2)


def get_real_scale_losses(losses: Losses, normalize_parameters) -> Losses:
    train_min, train_max = normalize_parameters
    for key in losses:
        if key not in ["r2"]:
            losses[key] = (
                losses[key] * (train_max["power"] - train_min["power"])
                + train_min["power"]
            )

    return losses
