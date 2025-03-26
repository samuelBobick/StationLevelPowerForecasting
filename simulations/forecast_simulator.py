import json
from typing import Literal, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from baseline_simulator import BaselineSimulator
from constants.global_parameters import (
    SMOOTH_POWER_FEATURES,
    VERBOSE_PREDICTIONS_NORMALIZED,
)
from constants.tariffs import MODIFIED_DC, TypeTariffName
from cvxpy.atoms.affine.hstack import Hstack
from slrp_ev_data.utils.scaling_utils import (
    retrieve_train_min_and_max,
)
from slrp_ev_ts_forecasting.default_parameters import SAVED_MODELS_PATH
from utils.utils_time_and_indexes import (
    convert_time_to_index,
    round_up_to_nearest_timestep,
)


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
        model_type: Literal["linear", "xgboost"] = "linear",
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

        self.model_type = model_type

        self.forecasting_models = {}
        for workday in [0, 1]:
            filename = f"{self.model_name}_{workday}.json"
            self.forecasting_models[workday] = self.load_model_parameters(filename)

        # get normalization parameters
        self.get_normalization_parameters(self.features_name, self.labels_name)

        self.lookahead = len(self.labels_name)

    def load_model_parameters(self, filename):
        with open(SAVED_MODELS_PATH / filename, "r") as json_file:
            model_params = json.load(json_file)

        self.check_model_parameters(model_params)

        self.features_name = model_params["feature_names"]
        self.labels_name = model_params["label_names"]

        if self.model_type == "linear":
            model_params["intercept"] = np.array(model_params["intercept"])
            model_params["coefficients"] = np.array(model_params["coefficients"])
        else:
            model_params = xgb.Booster()
            model_params.load_model(
                str(SAVED_MODELS_PATH / f"{filename[:-5]}_model.json")
            )

        return model_params

    def check_model_parameters(self, model_params):
        """Check if input parameters of the simulation correspond to
        the parameters the model was trained with"""

        model_scaling_mode = model_params["model_parameters"]["scaling_mode"]
        if model_scaling_mode != "normalize":
            raise ValueError(
                f"The scaling mode of your model {model_scaling_mode} is not yet supported. "
                "Only 'normalize' is supported for now. Please update your model."
            )

        model_var_dim = model_params["model_parameters"]["x_dim"]
        if model_var_dim != self.var_dim_constant:
            raise ValueError(
                f"The model was trained with a different var_dim_constant ({model_var_dim}) "
                f"than the one provided ({self.var_dim_constant}). "
                "Please update your model."
            )

        model_add_number_of_evses_available = model_params["model_parameters"][
            "add_number_of_evses_available"
        ]
        assert (
            model_add_number_of_evses_available
        ), "The model parameter add_number_of_evses_available should be True. Please update your model."

        model_add_number_of_sessions = model_params["model_parameters"][
            "add_number_of_sessions"
        ]
        assert (
            model_add_number_of_sessions
        ), "The model parameter add_number_of_sessions should be True. Please update your model."

        model_add_fraction_of_regular_sessions = model_params["model_parameters"][
            "add_fraction_of_regular_sessions"
        ]
        assert (
            not model_add_fraction_of_regular_sessions
        ), "The model parameter add_fraction_of_regular_sessions should be False. Please update your model."

        model_use_all_active_sessions = model_params["model_parameters"][
            "use_all_active_sessions"
        ]
        assert (
            model_use_all_active_sessions
        ), "The model parameter use_all_active_sessions should be True. Please update your model."

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
            elif name in norm_params_min.index:
                features_norm_parameters_min.append(norm_params_min[name])
                features_norm_parameters_max.append(norm_params_max[name])
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
            else:
                raise ValueError(f"Label {name} not recognized")
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

        if self.model_type == "linear":
            coefficients = model["coefficients"].squeeze()
            intercept = model["intercept"].squeeze()
            prediction = (coefficients @ normalized_features) + intercept
        else:
            dtest = xgb.DMatrix(
                pd.DataFrame(
                    data=normalized_features.reshape(1, len(normalized_features)),
                    columns=self.features_name,
                )
            )
            prediction = model.predict(
                dtest, iteration_range=(0, model.best_iteration + 1)
            ).squeeze()

        # the prediction can sometimes be negative, we need to make sure it is positive
        prediction = cp.maximum(prediction, 0)

        if verbose and VERBOSE_PREDICTIONS_NORMALIZED:
            print("Plotting scaled sample and prediction")
            self.visualize_samples(time, normalized_features.value, np.array(prediction.value))  # type: ignore

        reversed_prediction = (
            cp.multiply(
                prediction,
                (self.labels_norm_parameters_max - self.labels_norm_parameters_min),
            )
            + self.labels_norm_parameters_min
        )

        return reversed_prediction

    def smooth_profile(self, profile, window_size: int = 3):
        kernel = np.ones(window_size) / window_size

        smoothed_profile = cp.convolve(kernel, profile)

        # now we need to slice the profile because the size changed
        if window_size % 2 == 0:
            # even
            sliced_profile = smoothed_profile[
                (window_size // 2 - 1) : -(window_size // 2)
            ]
        else:
            # odd
            sliced_profile = smoothed_profile[(window_size // 2) : -(window_size // 2)]
        return sliced_profile

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

        # Add time of use in background
        whole_time_index = pd.concat([pd.Series(time_power), pd.Series(time_u)])
        number_timesteps_in_simulation = whole_time_index.shape[0]
        TOU_current_idx = convert_time_to_index(time, self.delta_t)
        TOU_data = pd.DataFrame(
            data={
                "TOU": list(self.TOU[TOU_current_idx:96])
                + list(self.TOU[:96])
                * ((number_timesteps_in_simulation - 96 + TOU_current_idx) // 96)
                + list(
                    self.TOU[
                        : (number_timesteps_in_simulation - 96 + TOU_current_idx) % 96
                    ]
                )
            },
            index=whole_time_index,
        )
        # Add a heatmap for TOU values as background
        fig.add_trace(
            go.Heatmap(
                x=TOU_data.index + pd.Timedelta(minutes=15 / 2),
                y=[0, 60_000],  # Adjust the heatmap to span from 0 to 60
                z=[TOU_data["TOU"]] * 2,  # Duplicate TOU values to match the y range
                colorscale="Greys",
                showscale=False,
                zmin=TOU_data["TOU"].min() * 0.5,
                zmax=TOU_data["TOU"].max() * 2,
                hovertemplate="TOU: %{z:.2f} cents/kWh<extra></extra>",
            )
        )

        # plot aggregated power
        fig.add_trace(
            go.Scatter(
                x=time_power,
                y=power,
                mode="lines",
                name="Aggregated Power until now",
                line=dict(color="blue"),
            )
        )
        if not SMOOTH_POWER_FEATURES:
            fig.add_trace(
                go.Scatter(
                    x=time_power,
                    y=self.smooth_profile(power).value,
                    mode="lines",
                    name="Aggregated Power until now (if smoothed)",
                    line=dict(dash="dot", color="blue"),
                )
            )

        # plot active sessions
        fig.add_trace(
            go.Scatter(
                x=time_u,
                y=u,
                mode="lines",
                name="Active Sessions Current Profile (u)",
                line=dict(dash="dash", color="red"),
            )
        )
        if not SMOOTH_POWER_FEATURES:
            fig.add_trace(
                go.Scatter(
                    x=time_u,
                    y=self.smooth_profile(u).value,
                    mode="lines",
                    name="Active Sessions Current Profile (u) (if smoothed)",
                    line=dict(dash="dot", color="red"),
                )
            )

        # plot prediction
        if type(prediction) in (
            float,
            int,
        ):  # if prediction is scalar, plot a point. Else, plot a time series.
            fig.add_trace(
                go.Scatter(
                    x=[time_u[0]],
                    y=prediction,
                    mode="markers",
                    name="Predicted Peak",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_u,
                    y=prediction,
                    mode="lines+markers",
                    name="Predicted Power",
                )
            )

        # show other features as text
        other_information = ""
        for i in other_indexes:
            other_information += f"{self.features_name[i]}: {sample[i]:.2f}<br>"

        fig.add_annotation(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=other_information,
            showarrow=False,
            align="right",
            font=dict(size=11),  # default is 12
        )

        fig.update_layout(
            title="Features and Prediction Visualization",
            xaxis_title="Time",
            yaxis_title="Power (W or scaled)",
        )

        # update yaxis limits with the min and max of the 3 power profiles
        fig.update_yaxes(
            range=[
                min(
                    power.min(),
                    u.min(),
                    prediction.min() if prediction is not None else 0,
                ),
                max(
                    power.max(),
                    u.max(),
                    prediction.max() if prediction is not None else 0,
                ),
            ]
        )
        fig.show()

    @property
    def model_name(self):
        if not hasattr(self, "_model_name") or self._model_name is None:
            self._model_name = None
            raise ValueError("Model name not set, it should be set in the child class")
        return self._model_name
