import numpy as np
import torch
import torch.nn as nn
from default_parameters import BETA
from sklearn.metrics import root_mean_squared_error  # type: ignore


def asymmetric_rmse(
    alpha: float, forecast: np.ndarray | list, real: np.ndarray | list
) -> float:
    forecast, real = np.array(forecast), np.array(real)
    weights = alpha ** (1 - np.sign(forecast - real))
    rwmse = root_mean_squared_error(forecast, real, sample_weight=weights)
    return rwmse


def asymmetric_rmse_detailed(y_true, y_pred, alpha: int) -> float:
    # Unused
    mse_loss = (y_true - y_pred) ** 2
    asymmetric_weight = alpha ** (1 - np.sign(y_pred - y_true))
    weighted_mse = asymmetric_weight * mse_loss
    rmse = np.sqrt(np.mean(weighted_mse))
    return rmse


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


def weighted_peaks_rmse(
    forecast: np.ndarray | list, real: np.ndarray | list, beta: float = BETA
) -> float:
    """
    Calculate the weighted peaks RMSE. This loss puts more weight on
    the peaks.
    """
    forecast, real = np.array(forecast), np.array(real)
    # rememnder that the values are scaled and should be between 0 and 1
    weights = 1 + beta * real
    # we need to add a 1 to avoid the weight being 0 when real is 0
    wprmse = root_mean_squared_error(forecast, real, sample_weight=weights)
    return wprmse


class WeightedPeaksRMSELoss(nn.Module):
    def __init__(self, beta: float = BETA):
        super(WeightedPeaksRMSELoss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        weights = 1 + self.beta * target
        loss = torch.sqrt(torch.mean(weights * mse_loss))
        return loss
