import os
from typing import Literal, Optional

import pandas as pd
import plotly.graph_objects as go
from baseline_simulator import BaselineSimulator
from constants.tariffs import MODIFIED_DC, TypeTariffName
from peak_forecast_simulator import PeakForecastSimulator
from smooth_dc_penalty_simulator import SmoothDCPenaltySimulator
from threshold_simulator import ThresholdSimulator
from timeseries_forecast_simulator import TimeseriesForecastSimulator
from utils import get_profit, get_session_results

TypeScenario = Literal[
    "all_scheduled",
    "all_regular",
    "standard",
    "smooth_dc_penalty",
    "threshold",
    "peak_forecast",
    "timeseries_forecast",
]


def filter_data(data, month, year, scenario: TypeScenario):
    """Helper function to filter the sessions dataframe

    Args:
        data (pd.DataFrame): dataframe of sessions
        month (int): month, as an integer. 1 = Jan, 2 = Feb, ...
        year (int): year
        scenario (string): if the scenario

    Returns:
        test_df (pd.DataFrame): filtered dataframe
    """

    test_df = data[
        (pd.to_datetime(data["connectTime"]).dt.year == year)
        & (pd.to_datetime(data["connectTime"]).dt.month == month)
    ]

    test_df = test_df[test_df["DurationHrs"] > 0.5]
    test_df = test_df[test_df["cumEnergy_Wh"] > 0]

    if scenario == "all_scheduled":
        test_df["choice"] = "SCHEDULED"
    elif scenario == "all_regular":
        test_df["choice"] = "REGULAR"
    else:
        print("INFO: Using historical choices")

    return test_df


def get_simulator(
    data,
    scenario: TypeScenario,
    var_dim_constant: int = 96,
    delta_t: float = 0.25,
    power_rate: float = 6.6,
    flexibility_constant: float = 0.57,
    tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
    custom_cost_dc: Optional[float] = MODIFIED_DC,
    step: float = 1,
    monte_carlo: bool = False,
    verbose: bool = False,
):
    """_summary_

    Args:
        scenario (_type_): _description_
    """
    # TODO this function returns different children of BaselineSimulator once we implement them (conditioned on scenario)
    if scenario in ["all_scheduled", "all_regular", "standard"]:
        return BaselineSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
        )
    elif scenario == "smooth_dc_penalty":
        return SmoothDCPenaltySimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
        )
    elif scenario == "threshold":
        return ThresholdSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
            step,
        )
    elif scenario == "peak_forecast":
        return PeakForecastSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
        )
    elif scenario == "timeseries_forecast":
        return TimeseriesForecastSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose,
        )
    else:
        raise ValueError(
            "Invalid scenario name in get_simulator. Please refer to TypeScenario."
        )


def generate_session_results(
    sim,
    month,
    results_file_name,
    summary_file_name,
    aggregate_power_profile_file_name,
    verbose=False,
    visualize=False,
):
    """Simulate. Save the simulation in results_file_name and append to summary file located at summary_file_name

    Args:
        sim (BaselineSimulator): simulator
        month (int): month, as an integer. 1 = Jan, 2 = Feb, ...
        results_file_name (string): filepath to store result dataframe
        summary_file_name (string): filepath of summary dataframe. If summary dataframe doesn't exist, creates a .csv file here.
        aggregate_power_profile_file_name (string): filepath of aggregate_power_profile dataframe. If summary dataframe doesn't exist, creates a .csv file here.
        verbose (bool, optional): _description_. Defaults to False. If true, prints summary information.
    """
    power_profiles, prices, hourly_prices = sim.simulate()
    session_results = get_session_results(
        sim.test_df, power_profiles, prices, sim.power_rate, sim.TOU, sim.delta_t
    )
    session_results.to_csv(results_file_name, index=False)

    charging_revenue, TOU_cost = get_profit(
        sim.test_df, power_profiles, prices, sim.delta_t, sim.TOU
    )

    demand_charge_kw = round(max(sim.aggregate_power_profile["power"]), 2)
    demand_charge_cents = round(sim.cost_dc * demand_charge_kw, 2)
    total_profit = round(charging_revenue - TOU_cost - demand_charge_cents, 2)
    charging_revenue = round(charging_revenue, 2)
    TOU_cost = round(TOU_cost, 2)
    energy_delivered = round(sum(session_results["energy_delivered"]), 2)

    row_data = {
        "Month": month,
        "Total Profit (cents)": total_profit,
        "Charging Revenue (cents)": charging_revenue,
        "TOU Cost (cents)": TOU_cost,
        "Demand Charge (cents)": demand_charge_cents,
        "Peak Power (kW)": demand_charge_kw,
        "Energy Delivered (kWh)": energy_delivered,
    }

    if os.path.exists(summary_file_name):
        summary_df = pd.read_csv(summary_file_name)
        summary_df = pd.concat(
            [summary_df, pd.DataFrame(row_data)],
            ignore_index=True,
        )
    else:
        columns = [
            "Month",
            "Total Profit (cents)",
            "Charging Revenue (cents)",
            "TOU Cost (cents)",
            "Demand Charge (cents)",
            "Peak Power (kW)",
            "Energy Delivered (kWh)",
        ]

        summary_df = pd.DataFrame([row_data], columns=columns)

    sim.aggregate_power_profile.to_csv(aggregate_power_profile_file_name, index=False)
    summary_df.to_csv(summary_file_name, index=False)

    if verbose:
        print("------------------------------------------------------------")
        for key in row_data:
            print(f"{key}: {row_data[key]}")
    if visualize:
        visualize_simulation(sim.aggregate_power_profile)


def visualize_simulation(aggregate_power_profile: pd.DataFrame) -> None:
    """
    Visualize the results of the simulation values.

    Args:
        aggregate_power_profile: dataframe with columns "date" and "power"
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aggregate_power_profile["date"],
            y=aggregate_power_profile["power"],
            mode="lines",
            name="Power Profile",
        )
    )

    fig.update_layout(
        title=f"Simulated Aggregate Power Profile: {min(aggregate_power_profile['date'].dt.date)} to {max(aggregate_power_profile['date'].dt.date)}",
        xaxis_title="Date",
        yaxis_title="Aggregate Station Power (kWh)",
    )
    fig.show()
