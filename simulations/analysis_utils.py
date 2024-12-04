import os
from typing import Optional

import pandas as pd
from baseline_simulator import BaselineSimulator
from constants.tariffs import TypeTariffName
from threshold_simulator import ThresholdSimulator
from utils import aggregate_power_profiles, get_profit, get_session_results


def filter_data(data, month, year, scenario):
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

    return test_df


def get_simulator(
    data,
    scenario,
    var_dim_constant: int = 96,
    delta_t: float = 0.25,
    power_rate: float = 6.6,
    flexibility_constant: float = 0.57,
    tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
    custom_cost_dc: Optional[float] = 500,
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


def generate_session_results(
    sim, month, results_file_name, summary_file_name, verbose=False
):
    """Simulate. Save the simulation in results_file_name and append to summary file located at summary_file_name

    Args:
        sim (BaselineSimulator): simulator
        month (int): month, as an integer. 1 = Jan, 2 = Feb, ...
        results_file_name (string): filepath to store result dataframe
        summary_file_name (string): filepath of summary dataframe. If summart dataframe doesn't exists, creates a .csv file here.
        verbose (bool, optional): _description_. Defaults to False. If true, prints summary information.
    """
    power_profiles, prices, hourly_prices = sim.simulate()
    session_results = get_session_results(
        sim.test_df, power_profiles, prices, sim.power_rate, sim.TOU, sim.delta_t
    )
    session_results.to_csv(results_file_name, index=False)

    agg_power_profile = aggregate_power_profiles(
        sim.test_df, power_profiles, sim.delta_t
    )
    charging_revenue, TOU_cost = get_profit(
        sim.test_df, power_profiles, prices, sim.delta_t, sim.TOU
    )

    demand_charge_kwh = round(max(agg_power_profile), 2)
    demand_charge_cents = round(sim.cost_dc * demand_charge_kwh, 2)
    total_profit = round(charging_revenue - TOU_cost - demand_charge_cents, 2)
    charging_revenue = round(charging_revenue, 2)
    TOU_cost = round(TOU_cost, 2)
    energy_delivered = round(sum(session_results["energy_delivered"]), 2)

    row = [
        month,
        total_profit,
        charging_revenue,
        TOU_cost,
        demand_charge_cents,
        demand_charge_kwh,
        energy_delivered,
        agg_power_profile,
    ]

    if os.path.exists(summary_file_name):
        summary_df = pd.read_csv(summary_file_name)
        summary_df = summary_df.append(row, ignore_index=True)
    else:
        columns = [
            "Month",
            "Total Profit (cents)",
            "Charging Revenue (cents)",
            "TOU Cost (cents)",
            "Demand Charge (cents)",
            "Peak Power (kWh)",
            "Energy Delivered (kW)",
            "Aggregate Power Profile (kW)",
        ]

        summary_df = pd.DataFrame([row], columns=columns)

    summary_df.to_csv(summary_file_name, index=False)

    if verbose:
        print("------------------------------------------------------------")
        print("Month", month)
        print("Total Profit", total_profit)
        print("Charging Revenue", charging_revenue)
        print("TOU Cost", TOU_cost)
        print("Demand Charge Costs (cents)", demand_charge_cents)
        print("Peak Power", demand_charge_kwh)
        print("Energy Delivered", energy_delivered)
