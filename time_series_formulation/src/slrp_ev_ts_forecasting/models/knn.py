from typing import Literal

import numpy as np
import pandas as pd
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from sklearn.neighbors import KNeighborsRegressor
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class KNN(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        n_neighbors=10,
        percentile=90,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            n_neighbors (int, optional): K in the KNN algorithm. Defaults to 10.
            percentile (int, optional): What percentile of the KNN we take. Defaults to 90.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            alpha=alpha,
            time_mode=time_mode,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
        )
        self.alpha = alpha
        self.time_mode = time_mode

        self.n_neighbors = n_neighbors
        self.percentile = percentile

    @property
    def model_str_name(self):
        return (
            f"KNN_neighbors{self.n_neighbors}_percentile{self.percentile}"
            + self.model_str_name_suffix
        )

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_mask: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        val_mask: pd.Series,
    ):
        knn_regressor = PercentileKNNRegressor(
            n_neighbors=self.n_neighbors, percentile=self.percentile
        )

        X_input = X_train[train_mask].drop(
            self.cols_to_drop_for_model,
            # [col for col in X_train.columns if not col.startswith("power")],
            axis=1,
        )
        y_input = y_train[train_mask]

        knn_regressor.fit(X_input, y_input)
        return knn_regressor

    def predict_model(self, model, X_test: pd.DataFrame):
        return model.predict(X_test)


class PercentileKNNRegressor:
    def __init__(self, n_neighbors=5, percentile=50):
        self.n_neighbors = n_neighbors
        self.percentile = percentile
        self.knn = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights="uniform", n_jobs=-1
        )

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        distances, indices = self.knn.kneighbors(X)
        nearest_neighbors_values = self.knn._y[indices]  # type: ignore

        nth_percentile_values = np.percentile(
            nearest_neighbors_values, self.percentile, axis=1
        )

        return nth_percentile_values
