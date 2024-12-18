import json
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from baseline_simulator import BaselineSimulator
from constants.tariffs import MODIFIED_DC, TypeTariffName
from cvxpy.atoms.affine.hstack import Hstack
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
    retrieve_train_min_and_max,
)
from slrp_ev_ts_forecasting.default_parameters import SAVED_MODELS_PATH
from utils import round_up_to_nearest_timestep


class PeakForecastSimulator(BaselineSimulator):
    def __init__(
        self,
        test_df,
        var_dim_constant: int = 96,
        delta_t: float = 0.25,
        power_rate: float = 6.6,
        flexibility_constant: float = 0.57,
        tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
        custom_cost_dc: Optional[float] = MODIFIED_DC,
        monte_carlo: bool = False,
        verbose: bool = False,
    ):
        """_summary_"""
        super().__init__(
            test_df,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
        )

        self.forecasting_models = {}
        for workday in [0, 1]:
            filename = f"LinearModel_SessionBased_PeakPrediction_{workday}.json"
            self.forecasting_models[workday] = self.load_model_parameters(filename)

        self.features_name = self.forecasting_models[0]["feature_names"]
        self.labels_name = self.forecasting_models[0]["label_names"]
        # get normalization parameters
        self.get_normalization_parameters(self.features_name, self.labels_name)

    def load_model_parameters(self, filename):
        with open(SAVED_MODELS_PATH / filename, "r") as json_file:
            params = json.load(json_file)
        params["intercept"] = np.array(params["intercept"])
        params["coefficients"] = np.array(params["coefficients"])
        return params

    def get_normalization_parameters(
        self, feature_names: list[str], label_names: list[str]
    ):
        norm_params_min, norm_params_max = retrieve_train_min_and_max("slrp-ev_new")
        norm_power_min = norm_params_min["power"]
        norm_power_max = norm_params_max["power"]
        norm_u_min = 0
        norm_u_max = SINGLE_EVSE_NORMALIZATION_PARAM

        features_norm_parameters_min = []
        features_norm_parameters_max = []
        for name in feature_names:
            if name.startswith("power"):
                features_norm_parameters_min.append(norm_power_min)
                features_norm_parameters_max.append(norm_power_max)
            elif name.startswith("u"):
                features_norm_parameters_min.append(norm_u_min)
                features_norm_parameters_max.append(norm_u_max)
            else:
                features_norm_parameters_min.append(0)
                features_norm_parameters_max.append(1)
        self.features_norm_parameters_min = np.array(features_norm_parameters_min)
        self.features_norm_parameters_max = np.array(features_norm_parameters_max)

        labels_norm_parameters_min = []
        labels_norm_parameters_max = []
        for name in label_names:
            if name.startswith("power") or name == "peak_power":
                labels_norm_parameters_min.append(norm_power_min)
                labels_norm_parameters_max.append(norm_power_max)
            elif name.startswith("u"):
                labels_norm_parameters_min.append(norm_u_min)
                labels_norm_parameters_max.append(norm_u_max)
            else:
                labels_norm_parameters_min.append(0)
                labels_norm_parameters_max.append(1)
        self.labels_norm_parameters_min = np.array(labels_norm_parameters_min)
        self.labels_norm_parameters_max = np.array(labels_norm_parameters_max)

    def make_prediction(
        self, features: np.ndarray | Hstack, workday: int
    ) -> cp.Expression:
        """Make a prediction using the linear model

        Args:
            features: 1D array with all the features. Make sure that \
                all the features are in the correct order (same \
                self.features_name). You can check the features order by visualizing \
                the samples with the function visualize_samples()
            workday: 1 if it is a workday, 0 if it is a non-workday.

        Returns:
            predictions
        """
        normalized_features = (features - self.features_norm_parameters_min) / (
            self.features_norm_parameters_max - self.features_norm_parameters_min
        )

        model = self.forecasting_models[workday]
        coefficients = model["coefficients"].squeeze()
        intercept = model["intercept"][0]

        prediction = (normalized_features @ coefficients) + intercept

        reversed_prediction = (
            prediction
            * (self.labels_norm_parameters_max - self.labels_norm_parameters_min)
            + self.labels_norm_parameters_min
        )

        return reversed_prediction

    def get_current_peak_sch(
        self, num_reg_user, num_sch_user, u, time
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

        Args:
            num_reg_user (int): number of regular users
            num_reg_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): timf of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        return self.get_current_peak(u, time)

    def get_current_peak_reg(
        self, num_reg_user, num_sch_user, u, time
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

            add the + 1 because we imagine that the new user is regular here
            the second term is basically the max power from the current scheduled users
            (without considering that the new user is scheduled)

        Args:
            num_reg_user (int): number of regular users
            num_reg_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): timf of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        return self.get_current_peak(u, time)

    def get_current_peak(self, u, time):
        """Make a forecast for the peak given the optimized power profile

        Args:
            u (cp.Variable): scheduled power profile
            time (pd.datetime): timf of optimization

        Returns:
            prediction of current peak given time, past power profile, and scheduled power_profile
        """
        rounded_current_time = round_up_to_nearest_timestep(time, self.delta_t)
        timesteps = pd.date_range(
            end=rounded_current_time - pd.Timedelta(minutes=15),
            periods=96,
            freq="15min",
        )
        historical_power_profile = self.aggregate_power_profile[
            self.aggregate_power_profile["date"].isin(timesteps)
        ]["power"].values

        s_in_day = 24 * 60 * 60  # number of seconds in a day
        s_in_week = 7 * s_in_day
        s_in_year = (365.2425) * s_in_day
        unix_time = time.timestamp()
        time_features = np.array(
            [
                np.sin(unix_time * (2 * np.pi / s_in_day)),
                np.cos(unix_time * (2 * np.pi / s_in_day)),
                np.sin(unix_time * (2 * np.pi / s_in_week)),
                np.cos(unix_time * (2 * np.pi / s_in_week)),
                np.sin(unix_time * (2 * np.pi / s_in_year)),
                np.cos(unix_time * (2 * np.pi / s_in_year)),
            ]
        )

        features = cp.hstack(
            [historical_power_profile, time_features, cp.reshape(u[:96], (96,))]
        )
        workday = int(time.dayofweek < 5)

        return self.make_prediction(features, workday)

    def visualize_samples(
        self, sample: np.ndarray, prediction: Optional[np.ndarray] = None
    ):
        power_indexes = []
        u_indexes = []
        for i, name in enumerate(self.features_name):
            if name.startswith("power"):
                power_indexes.append(i)
            elif name.startswith("u"):
                u_indexes.append(i)

        fig = go.Figure()

        power = sample[power_indexes]
        u = sample[u_indexes]
        time_power = np.arange(len(power)) * self.delta_t
        time_u = time_power + 24
        fig.add_trace(
            go.Scatter(
                x=time_power,
                y=power,
                mode="lines",
                name="Aggregated Power until now",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_u,
                y=u + power[-1],
                mode="lines",
                name="Next User Profile",
                line=dict(dash="dash"),
            )
        )
        if prediction:
            fig.add_trace(
                go.Scatter(
                    x=[time_u[0]],
                    y=prediction,
                    mode="markers",
                    name="Predicted Peak",
                )
            )
        fig.update_layout(
            title="Sample Visualization",
            xaxis_title="Time",
            yaxis_title="Power",
        )
        fig.show()
