from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from constants.tariffs import MODIFIED_DC, TypeTariffName
from forecast_simulator import ForecastSimulator
from utils import (
    get_aggregate_active_reg_future_profiles,
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
        self.forecast_historical_input_dim = 96

    def get_current_peak_sch(
        self, num_reg_user: int, num_sch_user: int, u: cp.Variable, time, verbose=False
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile
            time (pd.datetime): time of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        u_reshaped = cp.reshape(
            u, (u.shape[0] // self.var_dim_constant, self.var_dim_constant), order="C"
        )
        next_session_profile = cp.sum(u_reshaped, axis=0)

        aggregate_reg_profiles = get_aggregate_active_reg_future_profiles(
            self.test_df, time, self.power_profiles, self.delta_t
        )

        next_session_profile = next_session_profile + aggregate_reg_profiles

        return (
            self.get_current_peak(
                next_session_profile,
                num_reg_user + num_sch_user + 1,
                time,
                verbose=verbose,
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
            time (pd.datetime): time of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        e_need = get_total_e_need(row, self.delta_t, self.flexibility_constant)
        N_reg = int(
            e_need // self.power_rate
        )  # how many time steps would it take the user to charge if they chose regular?
        next_session_profile = np.array([self.power_rate] * N_reg + [0] * (96 - N_reg))

        u_sliced = u[
            self.var_dim_constant :
        ]  # to remove the part that considers the next user as scheduled
        if u_sliced.shape[0] > 0:
            u_reshaped = cp.reshape(u_sliced, ((u_sliced.shape[0]) // 96, 96))
            next_session_profile = next_session_profile + cp.sum(u_reshaped, axis=0)

        next_session_profile = (
            next_session_profile
            + get_aggregate_active_reg_future_profiles(
                self.test_df, time, self.power_profiles, self.delta_t
            )
        )

        # divide by 1000 to convert from W to kW
        return (
            self.get_current_peak(
                next_session_profile, num_reg_user + num_sch_user + 1, time
            )
            / 1000
        )

    def get_current_peak(
        self, aggregate_future_profile, num_active_sessions, time, verbose=False
    ) -> cp.Expression:
        """Make a forecast for the peak given the optimized power profile

        Args:
            next_scheduled_profile (cp.Variable): scheduled power profile
            num_active_sessions (int): number of active sessions
            time (pd.datetime): time of optimization

        Returns:
            prediction of current peak given time, past power profile, and scheduled power_profile
        """
        rounded_current_time = round_up_to_nearest_timestep(time, self.delta_t)
        historical_timesteps = pd.date_range(
            end=rounded_current_time - pd.Timedelta(minutes=15),
            periods=self.forecast_historical_input_dim,
            freq="15min",
        )

        historical_power_profile = self.aggregate_power_profile[
            self.aggregate_power_profile["date"].isin(historical_timesteps)
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

        time_in_15_minutes = time + pd.Timedelta(minutes=15)
        workday_in_15_minutes = int(time_in_15_minutes.dayofweek < 5)
        if time_in_15_minutes.date() in self.holidays.date:
            workday_in_15_minutes = 0

        time_tomorrow = time + pd.Timedelta(days=1)
        workday_tomorrow = int(time_tomorrow.dayofweek < 5)
        if time_tomorrow.date() in self.holidays.date:
            workday_tomorrow = 0

        features = cp.hstack(
            [
                historical_power_profile * 1000,
                workday_tomorrow,
                8,  # number EVSEs available (always 8 in SLRP-EV)
                time_features,
                aggregate_future_profile * 1000,
                num_active_sessions,
            ]
        )

        prediction = self.make_prediction(features, workday_in_15_minutes)

        if verbose:
            self.visualize_samples(time, sample=features.value, prediction=prediction.value)  # type: ignore

        return prediction
