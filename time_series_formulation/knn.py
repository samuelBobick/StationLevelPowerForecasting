import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


class KNN:

    def __init__(self, x_dim=16, lookahead=16, n_neighbors=10, percentile=90, alpha=2):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            n_neighbors (int, optional): K in the KNN algorithm. Defaults to 10.
            percentile (int, optional): What percentile of the KNN we take. Defaults to 90.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        self.lookahead = lookahead
        self.x_dim = x_dim

        self.alpha = alpha

        self.n_neighbors = n_neighbors
        self.percentile = percentile
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)


    def fit(self, train):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        X_train, y_train = self.get_X_y(train, self.lookahead)

        self.models = {}
        for t in [0, 4, 8, 12, 16, 20]:
            for w in [0, 1]:
                knn_regressor = PercentileKNNRegressor(n_neighbors=self.n_neighbors, percentile=self.percentile)
                mask = (X_train['time'] == t) & (X_train['workday'] == w)
                X_input = X_train[mask]['power']  
                y_input = y_train[mask]['power']  
                X_input = np.array([i for i in X_input])
                y_input = np.array([i for i in y_input])
                knn_regressor.fit(X_input, y_input)
                self.models[(t, w)] = knn_regressor




    def predict_single(self, X):
        """
        Args:
            X (iterable): One test point

        Returns:
            iterable: Forecasted power time series for the given trianing point
        """
        distances, indices = self.knn.kneighbors(X)
        nearest_neighbors_values = self.knn.y_test[indices]
        nth_percentile_values = np.percentile(nearest_neighbors_values, self.percentile, axis=1)
        return nth_percentile_values
    


    def predict(self, test):
        """
        Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        X_test, y_test = self.get_X_y(test, self.lookahead)

        mses = []
        wmses = []

        forecasts = []
        for index, row in X_test.iterrows():
            forecasts.append(self.models[(row['time'], row['workday'])].predict(np.array(row['power']).reshape(1, -1)))

        forecast = np.array([f[0] for f in forecasts]).flatten()
        real = np.array([a for a in y_test['power'].to_numpy()]).flatten()

        mse = mean_squared_error(forecast, real)

        weights = self.alpha ** (1 + np.sign(forecast - real))
        wmse = mean_squared_error(forecast, real, sample_weight=weights)

        mses.append(mse)
        wmses.append(wmse)

        return mse, wmse, forecast



    def get_X_y(self, df, lookahead):
        power = df['power'].to_numpy()
        workday = df['workday'].to_numpy()
        time = df['time'].to_numpy()
        
        power_chunks = [power[i:i+self.x_dim] for i in range(0, len(power) - 32, lookahead)]
        workday = [workday[i] for i in range(0, len(power) - 32, lookahead)]
        time = [time[i] for i in range(0, len(power) - 32, lookahead)]
        
        y = [power[i:i+lookahead] for i in range(32, len(power), lookahead)]

        X = pd.DataFrame(data={"power" : power_chunks, "workday" : workday, "time" : time})
        y = pd.DataFrame(data={"power" : y})

        X = X.reset_index()
        y = y.reset_index()

        return X, y
    


class PercentileKNNRegressor:
    def __init__(self, n_neighbors=5, percentile=50):
        self.n_neighbors = n_neighbors
        self.percentile = percentile
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        distances, indices = self.knn.kneighbors(X)
        nearest_neighbors_values = self.knn._y[indices]

        nth_percentile_values = np.percentile(nearest_neighbors_values, self.percentile, axis=1)

        return nth_percentile_values
    
