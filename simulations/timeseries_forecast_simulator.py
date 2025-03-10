from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from constants.tariffs import MODIFIED_DC, TypeTariffName
from forecast_simulator import ForecastSimulator
from slrp_ev_data.feature_engineering import engineer_time_features
from utils.utils import (
    aggregate_u_scheduled_profiles,
    get_aggregate_active_reg_future_profiles,
    get_next_reg_profile,
)
from utils.utils_time_and_indexes import (
    convert_time_to_index,
    round_up_to_nearest_timestep,
)


class TimeseriesForecastSimulator(ForecastSimulator):
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
        model_name: str = "LinearModel_SessionBased_WithNbSessions_WithAllActiveSessions8hr",
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
            model_name,
        )

        # Caches to save aggregte_future_profile (aka what we know about the future)
        # and the forecast itself (output of ML algorithm)
        self.prev_aggregate_u_for_forecast = np.zeros(self.var_dim_constant)
        self.forecast = np.zeros(self.var_dim_constant)

    def plot_reconstructed_timeseries(self, time, reconstructed_timeseries, forecast):
        time_forecast = pd.date_range(
            start=time, periods=len(reconstructed_timeseries), freq=f"{self.delta_t}H"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_forecast,
                y=reconstructed_timeseries,
                mode="lines",
                name="Reconstructed Timeseries",
                line=dict(dash="dash", color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_forecast,
                y=forecast,
                mode="lines",
                name="Forecast",
                line=dict(dash="dash", color="green"),
            )
        )
        fig.update_layout(
            title="Forecasted vs Reconstructed Timeseries for the next few hours",
            yaxis_title="Station Power Forecast (kW)",
        )

        fig.show()

    def get_current_peak_sch(
        self,
        num_reg_user: int,
        num_sch_user: int,
        u: cp.Variable,
        time,
        row,
        verbose=False,
    ) -> cp.Expression:
        """Helper fuction to get the peak, assuming the new user chooses scheduled

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): time of optimization
            row (pd.Series): row from the sessions DataFrame

        Returns:
            cp.Expression: current scheduled peak
        """
        next_reg_profile = get_next_reg_profile(
            row, self.delta_t, self.flexibility_constant, self.power_rate
        )

        reconstructed_timeseries = (
            self.forecast
            - next_reg_profile[: self.lookahead]
            - self.prev_aggregate_u_for_forecast[: self.lookahead]
            + aggregate_u_scheduled_profiles(u, self.lookahead)
        )

        if verbose:
            self.plot_reconstructed_timeseries(
                time, reconstructed_timeseries.value, self.forecast.value
            )

        return cp.max(reconstructed_timeseries)

    def get_current_peak_reg(
        self,
        num_reg_user: int,
        num_sch_user: int,
        u: cp.Variable,
        time,
        row,
        verbose=False,
    ) -> cp.Expression:
        """Helper fuction to get the peak, assuming the new user chooses regular

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): time of optimization
            row (pd.Series): row from the sessions DataFrame

        Returns:
            float: current regular peak
        """
        # to remove the part that considers the next user as scheduled
        u_sliced: cp.Variable = u[self.var_dim_constant :]  # type: ignore
        if u_sliced.shape[0] > 0:
            u_existing_scheduled = aggregate_u_scheduled_profiles(
                u_sliced, self.lookahead
            )
        else:
            u_existing_scheduled = np.zeros(self.lookahead)

        reconstructed_timeseries = (
            self.forecast
            - self.prev_aggregate_u_for_forecast[: self.lookahead]
            + u_existing_scheduled
        )

        if verbose:
            self.plot_reconstructed_timeseries(
                time, reconstructed_timeseries.value, self.forecast.value
            )

        return cp.max(reconstructed_timeseries)

    def get_timeseries_forecast(
        self,
        current_row,
        u_prev,
        prev_start_charge_time,
        prev_choice,
        num_active_sessions,
        current_start_charge_time,
        verbose=False,
    ):
        """Make a forecast for the next self.var_dim_constant timesteps, assuming the new user chooses regular.

        Args:
            current_row: row of sessions_df that is currently being optimized
            u: optimal power profile from the previous optimization. We haven't yet done the current optimization
            prev_start_charge_time: time of previous optimization, aka the start of u_prev
            prev_choice: previous user choice in the simulation, i.e. REGULAR or SCHEDULED
            time (pd.datetime): time of optimization
            num_active_sessions (int): number of active sessions

        Returns:
            prediction of current peak given time, past power profile, and scheduled power_profile
        """
        rounded_current_time = round_up_to_nearest_timestep(
            current_start_charge_time, self.delta_t
        )
        timesteps = pd.date_range(
            end=rounded_current_time - pd.Timedelta(minutes=15),
            periods=96,
            freq="15min",
        )

        aggregate_future_profile = get_next_reg_profile(
            current_row, self.delta_t, self.flexibility_constant, self.power_rate
        )

        if (
            u_prev is not None
        ):  # on the first iteration, u_prev will be None - there is no prior session
            if prev_choice == "REGULAR":
                u_prev_sliced = u_prev[self.var_dim_constant :]
            else:  # if scheduled, do not slice.
                u_prev_sliced = u_prev

            if u_prev_sliced.shape[0] > 0:
                u_prev_reshaped = aggregate_u_scheduled_profiles(
                    u_prev_sliced, self.var_dim_constant
                )
                timesteps_elapsed = convert_time_to_index(
                    current_start_charge_time - prev_start_charge_time, self.delta_t
                )

                timesteps_elapsed = min(timesteps_elapsed, self.var_dim_constant)

                u_prev_reshaped = np.pad(
                    u_prev_reshaped[timesteps_elapsed:].value,
                    (0, timesteps_elapsed),
                    mode="constant",
                    constant_values=0,
                )
            else:
                u_prev_reshaped = np.zeros(self.var_dim_constant)

            self.prev_aggregate_u_for_forecast = u_prev_reshaped
            aggregate_future_profile = aggregate_future_profile + u_prev_reshaped

        aggregate_future_profile = (
            aggregate_future_profile
            + get_aggregate_active_reg_future_profiles(
                self.test_df,
                current_start_charge_time,
                self.power_profiles,
                self.delta_t,
            )
        )

        historical_power_profile = self.aggregate_power_profile[
            self.aggregate_power_profile["date"].isin(timesteps)
        ]["power"].values

        # Sometimes at the beginning of the month, we must left pad to get ebough data to make a prediction
        historical_power_profile = np.pad(
            historical_power_profile,
            (self.var_dim_constant - historical_power_profile.size, 0),
            mode="constant",
        )

        # get time features
        time_features = pd.DataFrame(data=[current_start_charge_time], columns=["date"])
        engineer_time_features(time_features)
        # clean and put in the correct format
        time_features = time_features.drop(columns=["date"]).values.squeeze()

        time_tomorrow = current_start_charge_time + pd.Timedelta(hours=24)
        tomorrow_workday = int(time_tomorrow.dayofweek < 5)
        if time_tomorrow.date() in self.holidays.date:
            tomorrow_workday = 0

        time_in_15_min = current_start_charge_time + pd.Timedelta(minutes=15)
        time_in_15_min_workday = int(time_in_15_min.dayofweek < 5)
        if time_in_15_min.date() in self.holidays.date:
            time_in_15_min_workday = 0

        features = np.hstack(
            [
                historical_power_profile * 1000,
                tomorrow_workday,
                8,
                time_features,
                aggregate_future_profile[: self.lookahead] * 1000,
                num_active_sessions,
            ]
        )

        prediction = self.make_prediction(
            features, time_in_15_min_workday, current_start_charge_time
        )

        if verbose:
            self.visualize_samples(current_start_charge_time, features, prediction.value)  # type: ignore

        self.aggregate_future_profile_for_forecast = aggregate_future_profile
        self.forecast = prediction / 1000
