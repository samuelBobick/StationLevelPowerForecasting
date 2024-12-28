from typing import Literal

import numpy as np
import pandas as pd
from slrp_ev_ts_forecasting.compute_losses import Losses, compute_losses
from slrp_ev_ts_forecasting.default_parameters import TypeOptimizeLags
from slrp_ev_ts_forecasting.models.base import Base
from tqdm import tqdm


class RegressionBaseModel(Base):

    def __init__(
        self,
        x_dim: int,
        lookahead: int,
        alpha: float,
        time_mode: Literal["window", "cyclical"],
        optimize_lags: TypeOptimizeLags,
        get_val_data_from_shuffled_train: bool,
        session_based_mode: bool,
        peak_prediction: bool,
        add_number_of_sessions: bool,
        add_fraction_of_regular_sessions: bool,
        use_all_active_sessions: bool,
    ):
        """Base class for getting data, training and predicting regression models.

        Args:
            x_dim: How many past timesteps ahead we want to use as inputs.
            lookahead: How many timesteps ahead we want to predict,
            alpha: Underpredictions are penalized alpha times more than \
                overpredictions for weighted error metric. Defaults to 2.
            time_mode: Time mode for the model, can be cyclical (in which case) \
                the model will use cyclical features for the time of the day, \
                day of the week, and month of the year) or window (in which case \
                the model will use a window feature for the time of the day)
                and train a different model for each time window and workday.
            optimize_lags: Whether to optimize the lags for the model, i.e. look \
                for the best `x_dim` past timesteps in the `NUMBER_OF_DAYS_FOR_PACF` \
                past days, that have the highest partial autocorrelation with the \
                target variables.
            get_val_data_from_shuffled_train: Whether to get the \
                validation data from the shuffled train data. This can help \
                improving the algorithm's performance since there will more \
                recent data in the training set (otherwise, the most recent data \
                is in the val and test sets)
            session_based_mode: Whether to make a prediction for each session, or just \
                every `lookahead` timesteps.
            peak_prediction: Whether to make a peak prediction, i.e. predict the \
                maximum power in the next `lookahead` timesteps, rather than the \
                timeseries itself.
            add_number_of_sessions: Whether to add the number of sessions currently \
                active as a feature for session prediction
            add_fraction_of_regular_sessions: Whether to add the fraction of sessions \
                currently active that are regular as a feature for session prediction
        """
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
        )
        self.lookahead = lookahead
        self.x_dim = x_dim

        self.optimize_lags = optimize_lags
        self.session_based_mode = session_based_mode
        self.peak_prediction = peak_prediction

        self.alpha = alpha
        self.time_mode = time_mode
        if self.time_mode == "window":
            self.cols_to_drop_for_model = [
                "time_window",
                "workday",
            ]
        elif self.time_mode == "cyclical":
            self.cols_to_drop_for_model = [
                "workday",
                # "Year sin",
                # "Year cos",
            ]

    @property
    def model_str_name_suffix(self):
        return (
            ("_lagsOpti" if self.optimize_lags else "")
            + ("Short" if self.optimize_lags == "short_opt" else "")
            + ("Long" if self.optimize_lags == "long_opt" else "")
            + (
                "_SessionBased"
                + ("_PeakPrediction" if self.peak_prediction else "")
                + ("_WithNbSessions" if self.add_number_of_sessions else "")
                + ("_WithFracReg" if self.add_fraction_of_regular_sessions else "")
                + ("_WithAllActiveSessions" if self.use_all_active_sessions else "")
                if self.session_based_mode
                else ""
            )
        )

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        if self.optimize_lags:
            self.pacf_top_values = self.get_top_pacf_values(train)

        X_train, y_train = self.get_X_y(train, data_type="train", overlapping_windows=True, time_mode=self.time_mode)  # type: ignore
        self.update_seen_data(train)
        X_val, y_val = self.get_X_y(val, data_type="val", overlapping_windows=False, time_mode=self.time_mode)  # type: ignore
        if val is not None:
            self.update_seen_data(val)

        self.models = {}
        if self.time_mode == "window":
            for t_w in range(6):
                for w in [0, 1]:
                    train_mask = (X_train["time_window"] == t_w) & (
                        X_train["workday"] == w
                    )
                    val_mask = (X_val["time_window"] == t_w) & (X_val["workday"] == w)

                    self.models[(t_w, w)] = self.fit_model(
                        X_train, y_train, train_mask, X_val, y_val, val_mask
                    )
                    self.save_model(
                        self.models[(t_w, w)],
                        self.model_str_name + f"_{t_w}_{w}",
                    )
        elif self.time_mode == "cyclical":
            for w in [0, 1]:
                train_mask = X_train["workday"] == w
                val_mask = X_val["workday"] == w
                self.models[w] = self.fit_model(
                    X_train, y_train, train_mask, X_val, y_val, val_mask
                )
                self.save_model(self.models[w], self.model_str_name + f"_{w}")

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_mask: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        val_mask: pd.Series,
    ):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )

    def predict(self, test) -> tuple[Losses, pd.DataFrame]:
        """
        Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (Losses, DataFrame): Losses object, DataFrame of predictions with \
                "date" and multiple "power_x" columns for the predictions
        """
        X_test, y_test, y_dates = self.get_X_y(test, data_type="test", return_y_date=True, time_mode=self.time_mode)  # type: ignore

        forecasts = []

        for index, row in tqdm(X_test.iterrows(), desc="Predicting", total=len(X_test)):
            input = pd.DataFrame(
                [row.drop(self.cols_to_drop_for_model)]
            )  # .to_numpy().reshape(1, -1)
            if self.time_mode == "window":
                model = self.models[(row["time_window"], row["workday"])]
                forecasts.append(self.predict_model(model, input))

            elif self.time_mode == "cyclical":
                model = self.models[row["workday"]]
                forecasts.append(self.predict_model(model, input))

        forecast = np.array(forecasts).squeeze().flatten()
        real = y_test.to_numpy().flatten()
        losses = compute_losses(forecast, real, self.alpha)

        if not self.peak_prediction:
            reals = None
        else:
            reals = y_test.to_numpy()
        df_predictions = self.prepare_df_predictions(
            np.array(forecasts), y_dates, reals
        )
        return losses, df_predictions

    def predict_model(self, model, X_test: pd.DataFrame):
        raise NotImplementedError(
            "This method should be implemented by the child class"
        )
