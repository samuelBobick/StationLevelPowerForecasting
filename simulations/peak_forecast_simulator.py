from typing import Literal, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from constants.global_parameters import (
    SMOOTH_POWER_FEATURES,
    SMOOTH_WINDOW_SIZE,
    VERBOSE_PREDICTIONS_NORMALIZED,
)
from constants.tariffs import MODIFIED_DC, TypeTariffName
from forecast_simulator import ForecastSimulator
from slrp_ev_data.feature_engineering import engineer_time_features
from utils.utils import (
    aggregate_u_scheduled_profiles,
    get_aggregate_active_reg_future_profiles,
    get_next_reg_profile,
)
from utils.utils_time_and_indexes import round_up_to_nearest_timestep


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
        initial_running_peak: float = 0,
        monte_carlo: bool = False,
        verbose: bool = False,
        model_name: str = "LinearModel_SessionBased_PeakPrediction_WithNbSessions_WithAllActiveSessions",
        smooth_power_features: bool = SMOOTH_POWER_FEATURES,
        smooth_window_size: int = SMOOTH_WINDOW_SIZE,
        model_type: Literal["linear", "xgboost"] = "linear",
    ):
        self.model_type = model_type
        if self.model_type == "linear":
            self._model_name = "LinearModel_SessionBased_PeakPrediction_WithNbSessions_WithAllActiveSessions"
        else:
            raise ValueError(
                f"Model type {self.model_type} is not yet supported. Please choose linear."
            )

        """Child of ForecastSimulator"""
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
            model_type,
        )
        self.forecast_historical_input_dim = 96
        self.smooth_power_features = smooth_power_features
        self.smooth_window_size = smooth_window_size
        if self.smooth_power_features:
            print(
                f"INFO: Smoothing of the power features is enabled with a smoothing window size of {smooth_window_size}"
            )

    def get_current_peak_sch(
        self,
        num_reg_user: int,
        num_sch_user: int,
        u: cp.Variable,
        time,
        row,
        verbose=False,
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): power profile variable array of the scheduled users. \
                The 96 first elements are the next scheduled user, and the rest are the \
                existing scheduled users.
            time (pd.datetime): time of optimization
            row: UNUSED in peak forecast simulator

        Returns:
            cp.Expression: current scheduled peak
        """
        next_session_profile = aggregate_u_scheduled_profiles(u, self.var_dim_constant)

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
        self,
        num_reg_user: int,
        num_sch_user: int,
        u: cp.Variable,
        time,
        row,
        verbose=False,
    ) -> cp.Expression:
        """Helper fuction to get the peak, accounting for the optimized scheduled power profiles

            add the + 1 because we imagine that the new user is regular here
            the second term is basically the max power from the current scheduled users
            (without considering that the new user is scheduled)

        Args:
            num_reg_user (int): number of regular users
            num_sch_user (int): number of scheduled users
            u (cp.Variable): power profile variable array of the scheduled users. \
                The 96 first elements are the next scheduled user, and the rest are the \
                existing scheduled users.
            time (pd.datetime): time of optimization

        Returns:
            cp.Expression: current scheduled peak
        """
        next_session_profile = get_next_reg_profile(
            row, self.delta_t, self.flexibility_constant, self.power_rate
        )

        # to remove the part that considers the next user as scheduled
        u_sliced: cp.Variable = u[self.var_dim_constant :]  # type: ignore
        if u_sliced.shape[0] > 0:
            next_session_profile = (
                next_session_profile
                + aggregate_u_scheduled_profiles(u_sliced, self.var_dim_constant)
            )

        next_session_profile = (
            next_session_profile
            + get_aggregate_active_reg_future_profiles(
                self.test_df, time, self.power_profiles, self.delta_t
            )
        )

        # divide by 1000 to convert from W to kW
        return (
            self.get_current_peak(
                next_session_profile,
                num_reg_user + num_sch_user + 1,
                time,
                verbose=verbose,
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

        # Sometimes at the beginning of the month, we must left pad to get ebough data to make a prediction
        historical_power_profile = np.pad(
            historical_power_profile,
            (self.var_dim_constant - historical_power_profile.size, 0),
            mode="constant",
        )

        # get time features
        time_features = pd.DataFrame(data=[time], columns=["date"])
        engineer_time_features(time_features)
        # clean and put in the correct format
        time_features = time_features.drop(columns=["date"]).values.squeeze()

        time_in_15_minutes = time + pd.Timedelta(minutes=15)
        workday_in_15_minutes = int(time_in_15_minutes.dayofweek < 5)
        if time_in_15_minutes.date() in self.holidays.date:
            workday_in_15_minutes = 0

        time_tomorrow = time + pd.Timedelta(days=1)
        workday_tomorrow = int(time_tomorrow.dayofweek < 5)
        if time_tomorrow.date() in self.holidays.date:
            workday_tomorrow = 0

        if self.smooth_power_features:
            historical_power_profile = self.smooth_profile(
                historical_power_profile, self.smooth_window_size
            )
            aggregate_future_profile = self.smooth_profile(
                aggregate_future_profile, self.smooth_window_size
            )

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

        prediction = self.make_prediction(
            features, workday_in_15_minutes, time, verbose
        )

        if verbose and not VERBOSE_PREDICTIONS_NORMALIZED:
            self.visualize_samples(time, sample=features.value, prediction=prediction.value)  # type: ignore

        return prediction
