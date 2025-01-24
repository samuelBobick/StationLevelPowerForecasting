from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torcheval.metrics import R2Score

from slrp_ev_ts_forecasting.asymmetric_loss import (
    AsymmetricRMSELoss,
    WeightedPeaksRMSELoss,
    asymmetric_rmse_detailed,
    weighted_peaks_rmse,
)


class Losses(TypedDict):
    rmse: float
    wrmse: float
    mae: float
    wprmse: float
    r2: float
    error_std: float


def compute_losses(
    y_pred: np.ndarray | list, y_true: np.ndarray | list, alpha: float
) -> Losses:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wrmse = asymmetric_rmse_detailed(y_true, y_pred, alpha)
    mae = mean_absolute_error(y_true, y_pred)
    wprmse = weighted_peaks_rmse(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)
    error_std = np.std(np.array(y_pred) - np.array(y_true))
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2, error_std=error_std)  # type: ignore


def compute_torch_losses(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float
) -> Losses:
    rmse = torch.sqrt(nn.functional.mse_loss(y_pred, y_true, reduction="mean")).item()
    wrmse = AsymmetricRMSELoss(alpha)(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    wprmse = WeightedPeaksRMSELoss()(y_pred, y_true).item()
    # Both tensors are on the same device but we had a problem with the R2Score
    # so we can just move them to the cpu
    # print(f"y_pred device: {y_pred.device}")
    # print(f"y_true device: {y_true.device}")
    r2 = R2Score().update(y_pred.cpu(), y_true.cpu()).compute().item()
    error_std = torch.std(y_pred - y_true).item()
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2, error_std=error_std)  # type: ignore


def get_real_scale_losses(losses: Losses, normalize_parameters) -> Losses:
    train_min, train_max = normalize_parameters
    for key in losses:
        if key not in ["r2"]:
            losses[key] = (
                losses[key] * (train_max["power"] - train_min["power"])
                + train_min["power"]
            )

    return losses
