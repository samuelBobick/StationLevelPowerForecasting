import json
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from baseline_simulator import BaselineSimulator
from constants.global_parameters import VERBOSE_PREDICTIONS_NORMALIZED
from constants.tariffs import MODIFIED_DC, TypeTariffName
from cvxpy.atoms.affine.hstack import Hstack
from slrp_ev_data.normalization_and_standardization import (
    retrieve_train_min_and_max,
)
from slrp_ev_ts_forecasting.default_parameters import SAVED_MODELS_PATH
from utils.utils_time_and_indexes import round_up_to_nearest_timestep


class ForecastSimulator(BaselineSimulator):
    def __init__(
        self,
        test_df,
        var_dim_constant: int = 96,
        delta_t: float = 0.25,
        power_rate: float = 6.6,
        flexibility_constant: float = 0.57,
        tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
        custom_cost_dc: Optional[float] = MODIFIED_DC,
        initial_running_peak: float = 0,
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
            initial_running_peak,
            monte_carlo,
            verbose,
        )

        self.forecasting_models = {}
        for workday in [0, 1]:
            # TODO edit filename
            filename = f"LinearModel_SessionBased_PeakPrediction_WithNbSessions_WithAllActiveSessions_{workday}.json"
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
        # TODO: later we might need to put the time values between 0 and 1 (currently between -1 and 1)
        norm_params_min, norm_params_max = retrieve_train_min_and_max("slrp-ev_new")

        features_norm_parameters_min = []
        features_norm_parameters_max = []
        for name in feature_names:
            if name.startswith("power"):
                features_norm_parameters_min.append(norm_params_min["power"])
                features_norm_parameters_max.append(norm_params_max["power"])
            elif name == "number_of_evses_available":
                features_norm_parameters_min.append(
                    norm_params_min["number_of_evses_available"]
                )
                features_norm_parameters_max.append(
                    norm_params_max["number_of_evses_available"]
                )
            elif name.startswith("u"):
                features_norm_parameters_min.append(norm_params_min["power"])
                features_norm_parameters_max.append(norm_params_max["power"])
            else:
                features_norm_parameters_min.append(0)
                features_norm_parameters_max.append(1)
        self.features_norm_parameters_min = np.array(features_norm_parameters_min)
        self.features_norm_parameters_max = np.array(features_norm_parameters_max)

        labels_norm_parameters_min = []
        labels_norm_parameters_max = []
        for name in label_names:
            if name.startswith("power") or name == "peak_power":
                labels_norm_parameters_min.append(norm_params_min["power"])
                labels_norm_parameters_max.append(norm_params_max["power"])
            elif name.startswith("u"):
                labels_norm_parameters_min.append(norm_params_min["power"])
                labels_norm_parameters_max.append(norm_params_max["power"])
            else:
                labels_norm_parameters_min.append(0)
                labels_norm_parameters_max.append(1)
        self.labels_norm_parameters_min = np.array(labels_norm_parameters_min)
        self.labels_norm_parameters_max = np.array(labels_norm_parameters_max)

    def make_prediction(
        self,
        features: np.ndarray | Hstack,
        workday: int,
        time: pd.Timestamp,
        verbose: bool = False,
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

        # the prediction can sometimes be negative, we need to make sure it is positive
        # prediction = cp.maximum(prediction, 0)

        if verbose and VERBOSE_PREDICTIONS_NORMALIZED:
            self.visualize_samples(time, normalized_features.value, np.array(prediction.value))  # type: ignore

        reversed_prediction = (
            prediction
            * (self.labels_norm_parameters_max - self.labels_norm_parameters_min)
            + self.labels_norm_parameters_min
        )

        return reversed_prediction

    def visualize_samples(
        self,
        time: pd.Timestamp,
        sample: np.ndarray,
        prediction: Optional[np.ndarray] = None,
    ):
        power_indexes = []
        u_indexes = []
        other_indexes = []
        for i, name in enumerate(self.features_name):
            if name.startswith("power"):
                power_indexes.append(i)
            elif name.startswith("u"):
                u_indexes.append(i)
            else:
                other_indexes.append(i)

        fig = go.Figure()

        power = sample[power_indexes]
        u = sample[u_indexes]
        time = round_up_to_nearest_timestep(time, self.delta_t)
        time_power = pd.date_range(
            end=time - pd.Timedelta(minutes=15),
            periods=len(power),
            freq=f"{self.delta_t}H",
        )
        time_u = pd.date_range(start=time, periods=len(u), freq=f"{self.delta_t}H")
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
                y=u,
                mode="lines",
                name="Active Sessions Current Profile (u)",
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

        other_information = ""
        for i in other_indexes:
            other_information += f"{self.features_name[i]}: {sample[i]}<br>"

        fig.add_annotation(
            x=0.8,
            y=1,
            xref="paper",
            yref="paper",
            text=other_information,
            showarrow=False,
        )

        fig.update_layout(
            title="Sample and Peak Prediction Visualization",
            xaxis_title="Time",
            yaxis_title="Power (W or scaled)",
        )
        fig.show()
