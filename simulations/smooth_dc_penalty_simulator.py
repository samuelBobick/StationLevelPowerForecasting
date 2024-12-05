from typing import Optional

import cvxpy as cp
from baseline_simulator import BaselineSimulator
from constants.tariffs import MODIFIED_DC, TypeTariffName


class SmoothDCPenaltySimulator(BaselineSimulator):
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
        """_summary_

        Args:
            dc_penalty_smoothing (float, optional): smoothing parameter used in the \
                exponential penalty for demand charge. Defaults to 1/55, 55 being the \
                max power of the station (thus the max distance between the current \
                daily peak and the running peak).
        """
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

    def get_dc_penalty(self, current_daily_peak, running_monthly_peak) -> cp.Expression:
        """Apply a softplus penalty to the distance between the current daily peak and the running monthly peak."""
        x = current_daily_peak - running_monthly_peak
        return cp.Constant(self.cost_dc) * cp.log_sum_exp(cp.hstack([0, x]))
