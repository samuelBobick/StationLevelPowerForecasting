import warnings
from typing import Literal

import default_parameters
import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import mean_squared_error
from sktime.split import SlidingWindowSplitter

warnings.filterwarnings(
    "ignore", message=r".*does not have a custom `update` method implemented."
)


class SktimeBaseModel:
    def __init__(
        self,
        forecaster,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        alpha=default_parameters.ALPHA,
    ):

        self.alpha = alpha
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.time_mode = time_mode
        # Specifying forecasting horizon
        self.fh = np.arange(1, self.lookahead + 1)
        # Specifying the forecasting algorithm
        self.forecaster = forecaster
        # self.forecaster = ExponentialSmoothing(trend="add", seasonal="additive", sp=96)
        # self.forecaster = NaiveForecaster(strategy="mean", sp=96, window_length=96 * 7)

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> None:
        X_train, y_train = self.get_X_y(train)
        self.forecaster.fit(y_train, X=X_train)

        X_val, y_val = self.get_X_y(val)
        self.forecaster.update(y_val, X=X_val, update_params=True)

    def predict(
        self, test: pd.DataFrame
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        X_test, y_test = self.get_X_y(test)

        cv = SlidingWindowSplitter(
            fh=self.fh, window_length=self.x_dim, step_length=self.lookahead
        )

        y_pred = self.forecaster.update_predict(
            y_test, cv=cv, X=X_test, update_params=True
        )

        y_pred = y_pred.stack().reset_index(level=1, drop=True)
        y_test = y_test.loc[y_pred.index]

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        wrmse = asymmetric_rmse(y_test, y_pred, self.alpha)

        y_test_date = pd.to_datetime(y_pred.index).astype("int64") // 1e9

        return rmse, wrmse, y_pred.to_numpy(), y_test_date.to_numpy()

    def get_X_y(
        self,
        df: pd.DataFrame,
        overlapping_windows: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generates the dataset and features based on the input DataFrame."""
        data = df.copy()

        # Put date as index of the data. The index must have a frequency
        data["date"] = pd.to_datetime(data["date"], unit="s")
        data_freq = pd.infer_freq(data["date"])
        data = data.set_index("date")

        assert (
            (data == data.asfreq(data_freq)).all().all()
        ), "Data is not evenly spaced or contains missing intervals"

        data = data.asfreq(data_freq)

        cols_to_keep_as_features = ["power"]

        if self.time_mode == "cyclical":
            cols_to_keep_as_features += [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Year sin",
                "Year cos",
            ]
        elif self.time_mode == "window":
            cols_to_keep_as_features += ["time_window", "workday"]

        X = data[cols_to_keep_as_features]
        y = data["power"]

        return X, y


def asymmetric_rmse(y_true, y_pred, alpha: int) -> float:
    mse_loss = (y_true - y_pred) ** 2
    asymmetric_weight = alpha ** (1 - np.sign(y_pred - y_true))
    weighted_mse = asymmetric_weight * mse_loss
    rmse = np.sqrt(np.mean(weighted_mse))
    return rmse
