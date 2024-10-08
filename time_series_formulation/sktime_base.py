import warnings
from typing import Literal

import default_parameters
import numpy as np
import pandas as pd
from asymmetric_loss import asymmetric_rmse
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
        include_exogenous: bool = True,
        reduce_data_frequency: bool = False,
        refit_model_before_predictions: bool = False,
        start_data_date: str = "2020",
    ):
        """
        Initializes the SktimeBase model.
        Args:
            forecaster: The forecasting algorithm to be used. (one of sktime's forecasting algorithms see [here](https://www.sktime.net/en/stable/examples/01_forecasting.html#2.-Forecasters-in-sktime---lookup,-properties,-main-families))
            x_dim: The dimensionality of the input data (the number of steps to look back in the past). Defaults to default_parameters.X_DIM.
            lookahead: The number of steps to forecast into the future. Defaults to default_parameters.LOOKAHEAD.
            time_mode: The mode for handling time. Can be "window" or "cyclical". Defaults to default_parameters.TIME_MODE.
            alpha: The alpha value for asymmetric loss function. Defaults to default_parameters.ALPHA.
            include_exogenous: Whether to include exogenous variables in the model. This means that when making predictions for time [t, t+1, ..., t+lookahead], the model will use the exogenous variables at time [t, t+1, ..., t+lookahead]. Defaults to True.
            reduce_data_frequency: Whether to reduce the data frequency to a 1 hour frequency. Defaults to False.
            refit_model_before_predictions: Whether to refit the model before making predictions. Defaults to False because we do not refit the other models with new data (FFNN, LSTM, KNN, etc.).
            start_data_date: The start date to truncate the data at. Defaults to "2020" (equivalent to no truncation).
        """

        self.alpha = alpha
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.time_mode = time_mode

        # The following parameters are meant to make the model faster
        self.include_exogenous = include_exogenous
        self.reduce_data_frequency = reduce_data_frequency
        self.frequency_reduction_factor = 1  # initialize this to 1 for the case where we don't reduce the data frequency
        self.refit_model_before_predictions = refit_model_before_predictions
        self.start_data_date = start_data_date

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
        X_val, y_val = self.get_X_y(val)
        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])
        # Specifying forecasting horizon
        self.fh = np.arange(
            1, int(self.lookahead / self.frequency_reduction_factor) + 1
        )

        if self.reduce_data_frequency:
            # TODO: keep this only if reducing size
            # Drop rows with duplicate indexes
            X = X[~X.index.duplicated(keep="first")]
            y = y[~y.index.duplicated(keep="first")]
            assert (
                (X == X.asfreq("1h")).all().all()
            ), "Data is not evenly spaced or contains missing intervals"
            X = X.asfreq("1h")
            y = y.asfreq("1h")

        if self.include_exogenous:
            self.forecaster.fit(y, X=X, fh=self.fh)
        else:
            self.forecaster.fit(y, fh=self.fh)

        # self.forecaster.update(y_val, X=X_val, update_params=True)

        print(f"Fitted parameters are: {self.forecaster.get_fitted_params()}")

    def predict(
        self, test: pd.DataFrame
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        X_test, y_test = self.get_X_y(test)

        # Reduce the number of predictions if the algorithm is too slow
        # if self.frequency_reduction_factor:
        #     number_of_days_to_keep = 60
        #     X_test = X_test.iloc[: number_of_days_to_keep * 24]
        #     y_test = y_test.iloc[: number_of_days_to_keep * 24]

        cv = SlidingWindowSplitter(
            fh=self.fh,
            window_length=int(self.x_dim / self.frequency_reduction_factor),
            step_length=int(self.lookahead / self.frequency_reduction_factor),
        )

        if self.include_exogenous:
            y_pred_raw = self.forecaster.update_predict(
                y_test,
                cv=cv,
                X=X_test,
                update_params=self.refit_model_before_predictions,
                reset_forecaster=False,
            )
        else:
            y_pred_raw = self.forecaster.update_predict(
                y_test,
                cv=cv,
                update_params=self.refit_model_before_predictions,
                reset_forecaster=False,
            )

        y_pred = y_pred_raw.stack().reset_index(level=1, drop=True)
        if len(y_pred.shape) > 1:
            y_pred = y_pred_raw.stack().stack().reset_index(level=[1, 2], drop=True)
        y_test = y_test.loc[y_pred.index]

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        wrmse = asymmetric_rmse(self.alpha, y_pred, y_test)

        y_test_date = pd.to_datetime(y_pred.index).astype("int64") // 1e9

        return rmse, wrmse, y_pred.to_numpy(), y_test_date.to_numpy()

    def predict_short(
        self, test: pd.DataFrame, number_of_predictions: int = 1
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        X_test, y_test = self.get_X_y(test)

        # cv = SlidingWindowSplitter(
        #     fh=self.fh, window_length=self.x_dim, step_length=self.lookahead
        # )

        # # Access the first split
        # first_split = next(cv.split(X_test))
        # first_X_test = X_test.iloc[first_split]

        if self.include_exogenous:
            y_pred = self.forecaster.predict(fh=self.fh, X=X_test.iloc[self.fh])
        else:
            y_pred = self.forecaster.predict(fh=self.fh)

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

        if self.reduce_data_frequency:
            # reduce sample size
            data = data.loc[data.index >= self.start_data_date]
            # resample to 1 hour data
            data = data.resample("1h").mean()
            if data_freq == "15min":
                self.frequency_reduction_factor = 4
            elif data_freq == "5min":
                self.frequency_reduction_factor = 12
            else:
                raise ValueError(f"Data frequency of {data_freq} is not supported yet")

        cols_to_keep_as_features = []

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
