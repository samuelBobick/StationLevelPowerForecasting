from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from constants.tariffs import MODIFIED_DC, TypeTariffName
from forecast_simulator import ForecastSimulator
from utils import (
    get_aggregate_reg_profiles,
    get_total_e_need,
    round_up_to_nearest_timestep,
)


class PeakForecastSimulator(ForecastSimulator):
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

    def get_current_peak_sch(
        self, num_reg_user: int, num_sch_user: int, u: cp.Variable, time, row
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): timf of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        next_session_profile = cp.reshape(u[:96], (96,))

        u_reshaped = cp.reshape(
            u, (u.shape[0] // self.var_dim_constant, self.var_dim_constant)
        )
        next_session_profile = cp.sum(u_reshaped, axis=0)
        aggregate_reg_profiles = get_aggregate_reg_profiles(
            self.test_df, time, self.power_profiles, self.delta_t
        )

        next_session_profile = next_session_profile + aggregate_reg_profiles

        return (
            self.get_current_peak(
                next_session_profile, num_reg_user + num_sch_user - 1, time
            )
            / 1000
        )

    def get_current_peak_reg(
        self, num_reg_user: int, num_sch_user: int, u: cp.Variable, time, row
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

            add the + 1 because we imagine that the new user is regular here
            the second term is basically the max power from the current scheduled users
            (without considering that the new user is scheduled)

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): timf of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        e_need = get_total_e_need(row, self.delta_t, self.flexibility_constant)
        N_reg = int(
            e_need // self.power_rate
        )  # how many time steps would it take the user to charge if they chose regular?
        next_session_profile = np.array([self.power_rate] * N_reg + [0] * (96 - N_reg))

        u_sliced = u[96:]
        if u_sliced.shape[0] > 0:
            u_reshaped = cp.reshape(u_sliced, ((u_sliced.shape[0]) // 96, 96))
            next_session_profile = next_session_profile + cp.sum(u_reshaped, axis=0)

        next_session_profile = next_session_profile + get_aggregate_reg_profiles(
            self.test_df, time, self.power_profiles, self.delta_t
        )

        # divide by 1000 to convert from W to kW
        return (
            self.get_current_peak(
                next_session_profile, num_reg_user + num_sch_user - 1, time
            )
            / 1000
        )

    def get_current_peak(
        self, next_session_profile, num_active_sessions, time, verbose=False
    ) -> cp.Expression:
        """Make a forecast for the peak given the optimized power profile

        Args:
            next_scheduled_profile (cp.Variable): scheduled power profile
            time (pd.datetime): time of optimization
            num_active_sessions (int): number of active sessions

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

        workday = int(time.dayofweek < 5)
        if time.date() in self.holidays.date:
            workday = 0

        features = cp.hstack(
            [
                historical_power_profile * 1000,
                workday,
                8,  # number EVSEs available (always 8 in SLRP-EV) TODO standardize?
                time_features,
                next_session_profile * 1000,
                num_active_sessions,
            ]
        )

        # TODO Thibaud can you review this and make sure I am doing it right? Also make it less hardcoded
        features = (features - self.features_norm_parameters_min) / (
            self.features_norm_parameters_max - self.features_norm_parameters_min
        )

        prediction = self.make_prediction(features, workday)

        if verbose:
            self.visualize_samples(features.value, prediction.value)  # type: ignore

        return prediction

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
