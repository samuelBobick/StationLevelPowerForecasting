import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import warnings

# Suppress all warnings
# warnings.filterwarnings("ignore")


class NeuralNet:

    def __init__(
        self,
        x_dim=16,
        lookahead=16,
        alpha=2,
        input_size=22,
        hidden_size=64,
        output_size=16,
        num_hidden_layers=2,
        activation=nn.ReLU(),
        learning_rate=0.01,
        epochs=2000,
    ):
        """
        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (float, optional): . Defaults to 2.

            Neural Net Parameters:
            input_size (int, optional): Defaults to 22. TODO: Should be 16 + 5 + 1, where 16 is the x_dim, 5 is the one-hot encoding of the time of day, and 1 is the workday.
            hidden_size (int, optional): Defaults to 64.
            output_size (int, optional): Defaults to 16. TODO: Should be equal to lookahead.
            num_hidden_layers (int, optional): Defaults to 2.
            activation (_type_, optional): Defaults to nn.ReLU().
            learning_rate (float, optional): Defaults to 0.01.
            epochs (int, optional): Defaults to 1000.
        """
        self.x_dim = x_dim
        self.lookahead = lookahead
        self.alpha = alpha
        self.epochs = epochs

        self.ffnn = FFNN(
            input_size, hidden_size, output_size, num_hidden_layers, activation
        )
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

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            self.ffnn.train()
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.ffnn(X_train_tensor)

            loss = self.criterion(y_pred.squeeze(), y_train_tensor)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Log training loss
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")

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
        rmse = np.sqrt(
            self.criterion.forward(
                torch.tensor(y_test_tensor, dtype=torch.float32),
                torch.tensor(y_pred_test, dtype=torch.float32),
            ).item()
        )

        weighted_criterion = AsymmetricRMSELoss(alpha=2)
        wrmse = np.sqrt(
            weighted_criterion.forward(
                torch.tensor(y_test_tensor, dtype=torch.float32),
                torch.tensor(y_pred_test, dtype=torch.float32),
            ).item()
        )

        # Flatten the list to shape 971*16
        y_pred_test_flat = [item for sublist in y_pred_test for item in sublist]
        return rmse, wrmse, y_pred_test_flat

    def get_X_y(self, df, lookahead):
        """

        Args:
            df (DataFrame): DataFrame with columns "power", "workday", and "time"
            x_dim (int): How many past timesteps ahead we want to use as inputs.

        Returns:
            tuple (X - a numpy array of features, y - a numpy array of power readings)
        """
        df = df.copy()
        # This algorithm only works with data starting at the beginning of an interval
        # (hour= 0 or 4 or 8 , etc.).
        # To make sure we start at the beginning of an interval, let's just start at the
        # beginning of a day
        df = df[
            df["date"]
            >= (pd.to_datetime(df.iloc[0]["date"].date()) + pd.Timedelta(days=1))
        ]

        power = df["power"].to_numpy()
        workday = df["workday"].to_numpy()
        time = (df["date"].dt.hour + df["date"].dt.minute / 60).to_numpy()

        # to make sure the last y interval can have the lookahead size, we need to compute
        # the final possible window interval
        final_possible_window_index = len(power) - (len(power) % lookahead)
        power_chunks = [
            power[i : i + self.x_dim]
            for i in range(0, final_possible_window_index - self.x_dim, lookahead)
        ]
        workday = [
            workday[i]
            for i in range(self.x_dim, final_possible_window_index, lookahead)
        ]
        time = [
            time[i] for i in range(self.x_dim, final_possible_window_index, lookahead)
        ]

        y = [
            power[i : i + lookahead]
            for i in range(self.x_dim, final_possible_window_index, lookahead)
        ]

        X = pd.DataFrame(data={"power": power_chunks, "workday": workday, "time": time})
        y = pd.DataFrame(data={"power": y})

        X = X.reset_index()
        y = y.reset_index()

        X_lst = []

        for index, row in X.iterrows():
            time_ohe = [0, 0, 0, 0, 0]
            # if row["time"] < 20: TODO: why do we have this line? -1 added below
            time_ohe[int(row["time"] / 4) - 1] = 1
            X_lst.append(list(row["power"]) + [row["workday"]] + time_ohe)

        X = np.array(X_lst)

        y_lst = []
        for index, row in y.iterrows():
            y_lst.append(list(row["power"]))
        y = np.array(y_lst)

        return X, y


class FFNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden_layers, activation
    ):
        super(FFNN, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)]
        )
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
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        residual = input - target
        mask = residual <= 0  # mask for underpredictions
        loss = torch.sqrt(
            torch.mean((1 + (self.multiplier - 1) * mask.float()) * mse_loss)
        )
        return loss
