from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from constants.tariffs import MODIFIED_DC, TypeTariffName
from forecast_simulator import ForecastSimulator
from utils import get_next_reg_profile, round_up_to_nearest_timestep


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
        timeseries = (
            self.get_timeseries(row, num_reg_user + num_sch_user - 1, time) / 1000
        )
        next_reg_profile = get_next_reg_profile(
            row, self.delta_t, self.flexibility_constant, self.power_rate
        )

        return cp.max(
            timeseries
            - next_reg_profile
            + cp.reshape(u[: self.var_dim_constant], (self.var_dim_constant,))
        )

    def get_current_peak_reg(
        self, num_reg_user: int, num_sch_user: int, u: cp.Variable, time, row
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
        # Wrap with cp.constant so we can call .value on it
        return cp.Constant(
            max(self.get_timeseries(row, num_reg_user + num_sch_user - 1, time) / 1000)
        )

    def get_timeseries(
        self, current_row, num_active_sessions, time, verbose=False
    ) -> cp.Expression:
        """Make a forecast for the next self.var_dim_constant timesteps, assuming the new user chooses regular.

        Args:
            current_row: row of sessions_df that is currently being optimized
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
        next_reg_profile = get_next_reg_profile(
            current_row, self.delta_t, self.flexibility_constant, self.power_rate
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
                next_reg_profile * 1000,
                num_active_sessions,
            ]
        )

        # TODO plug in model from Thibaud and delete my dummy prediction
        prediction = self.make_prediction(features, workday)
        prediction = np.zeros(self.var_dim_constant)

        if verbose:
            self.visualize_samples(features.value, prediction.value)  # type: ignore

        return prediction

    def visualize_samples(
        self, sample: np.ndarray, prediction: Optional[np.ndarray] = None
    ):
        # TODO
        pass

    def get_timeseries_forecast(self, current_row, num_active_sessions, time, verbose=False):
        """Make a forecast for the next self.var_dim_constant timesteps, assuming the new user chooses regular.

        Args:
            current_row: row of sessions_df that is currently being optimized
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

        next_reg_profile = get_next_reg_profile(
            current_row, self.delta_t, self.flexibility_constant, self.power_rate
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

        time_tomorrow = time + pd.Timedelta(hours=24)
        tomorrow_workday = int(time_tomorrow.dayofweek < 5)
        if time_tomorrow.date() in self.holidays.date:
            tomorrow_workday = 0

        time_in_15_min = time + pd.Timedelta(minutes=15)
        time_in_15_min_workday = int(time_in_15_min.dayofweek < 5)
        if time_in_15_min.date() in self.holidays.date:
            time_in_15_min_workday = 0

        features = cp.hstack(
            [
                historical_power_profile * 1000,
                tomorrow_workday,
                8, # TODO normalize
                time_features,
                (aggregate_future_profile + next_reg_profile) * 1000,
                num_active_sessions,
            ]
        )

        # TODO plug in model from Thibaud and delete my dummy prediction
        prediction = self.make_prediction(features, time_in_15_min_workday)
        prediction = np.zeros(self.var_dim_constant)

        if verbose:
            self.visualize_samples(features.value, prediction.value)  # type: ignore

        return prediction
