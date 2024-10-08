from typing import Literal

import default_parameters
import numpy as np
from regression_base import RegressionBaseModel
from sklearn.neighbors import KNeighborsRegressor


class KNN(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        n_neighbors=10,
        percentile=90,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
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
            x_dim=x_dim, lookahead=lookahead, alpha=alpha, time_mode=time_mode
        )
        self.alpha = alpha
        self.time_mode = time_mode

        self.n_neighbors = n_neighbors
        self.percentile = percentile

    def fit_model(self, X_train, y_train, data_mask):
        knn_regressor = PercentileKNNRegressor(
            n_neighbors=self.n_neighbors, percentile=self.percentile
        )

        X_input = (
            X_train[data_mask]
            .drop(
                self.cols_to_drop_for_model,
                # [col for col in X_train.columns if not col.startswith("power")],
                axis=1,
            )
            .to_numpy()
        )
        y_input = y_train[data_mask].to_numpy()

        knn_regressor.fit(X_input, y_input)
        return knn_regressor


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
        nearest_neighbors_values = self.knn._y[indices]

        nth_percentile_values = np.percentile(
            nearest_neighbors_values, self.percentile, axis=1
        )

        return nth_percentile_values
