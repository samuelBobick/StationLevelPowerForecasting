import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import product

import warnings
# Suppress all warnings
# warnings.filterwarnings("ignore")


class NeuralNet:

    def __init__(self, x_dim=16, lookahead=16, alpha=2, input_size=22, hidden_size=64, output_size=16, num_hidden_layers=2, activation=nn.ReLU(), learning_rate=0.01, epochs=1000):
        """
        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (float, optional): . Defaults to 2.

            Neural Net Parameters:
            input_size (int, optional): Defaults to 22.
            hidden_size (int, optional): Defaults to 64.
            output_size (int, optional): Defaults to 16.
            num_hidden_layers (int, optional): Defaults to 2.
            activation (_type_, optional): Defaults to nn.ReLU().
            learning_rate (float, optional): Defaults to 0.01.
            epochs (int, optional): Defaults to 1000.
        """
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.alpha = alpha
        self.epochs = epochs

        self.ffnn = FFNN(input_size, hidden_size, output_size, num_hidden_layers, activation)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.ffnn.parameters(), lr=learning_rate)

    def fit(self, train):
        """Train self.ffnn

        Args:
            train (DataFrame): Training dataframe train with columns "power", "workday", and "time"
        """
        X_train, y_train = self.get_X_y(train, self.lookahead)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        for epoch in np.arange(self.epochs):
            self.ffnn.train()
            self.optimizer.zero_grad()
            y_pred = self.ffnn(X_train_tensor)
            loss = self.criterion(y_pred.squeeze(), y_train_tensor)
            loss.backward()
            self.optimizer.step()


    def predict(self, test):
        """Given a pandas DataFrame test with a power column, returns error metrics and list of predictions

        Args:
            test (DataFrame): test DataFrame with columns "power", "workday", and "time"

        Returns:
            tuple (float, float, list): RMSE, weighted RMSE, array of predictions
        """
        X_test, y_test = self.get_X_y(test, self.lookahead)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        self.ffnn.eval()
        y_pred_test = self.ffnn(X_test_tensor).detach().numpy().squeeze()
        rmse = np.sqrt(self.criterion.forward(torch.tensor(y_test_tensor, dtype=torch.float32), torch.tensor(y_pred_test, dtype=torch.float32)).item())

        criterion = AsymmetricRMSELoss(alpha=2)
        wrmse = np.sqrt(criterion.forward(torch.tensor(y_test_tensor, dtype=torch.float32), torch.tensor(y_pred_test, dtype=torch.float32)).item())

        return rmse, wrmse, y_pred_test

    def get_X_y(self, df, x_dim):
        """

        Args:
            df (DataFrame): DataFrame with columns "power", "workday", and "time"
            x_dim (int): How many past timesteps ahead we want to use as inputs.

        Returns:
            tuple (X - a numpy array of features, y - a numpy array of power readings)
        """
        power = df['power'].to_numpy()
        workday = df['workday'].to_numpy()
        time = df['time'].to_numpy()
        
        power_chunks = [power[i:i+x_dim] for i in range(0, len(power) - x_dim, x_dim)]
        workday = [workday[i] for i in range(0, len(power) - x_dim, x_dim)]
        time = [time[i] for i in range(0, len(power) - x_dim, x_dim)]
        
        y = [power[i:i+x_dim] for i in range(x_dim, len(power), x_dim)]

        X = pd.DataFrame(data={"power" : power_chunks, "workday" : workday, "time" : time})
        y = pd.DataFrame(data={"power" : y})

        X = X.reset_index()
        y = y.reset_index()


        X_lst = []

        for index, row in X.iterrows():
            time_ohe = [0, 0, 0, 0, 0]
            if row['time'] < 20:
                time_ohe[int(row['time'] / 4)] = 1
            X_lst.append(list(row['power']) + [row['workday']] + time_ohe)

        X = np.array(X_lst)

        y_lst = []
        for index, row in y.iterrows():
            y_lst.append(list(row['power']))

        y = np.array(y_lst)

        return X, y


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, activation):
        super(FFNN, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers-1)])
        self.activation = activation
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = torch.relu(self.fc_out(x))
        return x


class AsymmetricRMSELoss(nn.Module):
    def __init__(self, alpha):
        super(AsymmetricRMSELoss, self).__init__()
        self.multiplier = alpha**2

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction='none')
        residual = input - target
        mask = residual <= 0  # mask for underpredictions
        loss = torch.sqrt(torch.mean((1 + (self.multiplier - 1) * mask.float()) * mse_loss))
        return loss

    

