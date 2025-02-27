import os
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
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
        verbose (bool, optional): If true, prints summary information of the results. Defaults to False.
        visualize (bool, optional): If true, visualizes the results. Defaults to False.
    """
    power_profiles, prices, hourly_prices, user_computed_data_for_visualization = (
        sim.simulate()
    )
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
        visualize_simulation(
            sim.aggregate_power_profile,
            power_profiles,
            user_computed_data_for_visualization,
            prices,
        )


def visualize_simulation(
    aggregate_power_profile: pd.DataFrame,
    power_profiles: dict,
    user_computed_data_for_visualization: dict,
    prices: dict,
) -> None:
    """
    Visualize the results of the simulation values. the x axis is time

    Args:
        aggregate_power_profile: dataframe with columns "date" and "power"
    """
    df_user_computed_data_for_visualization = pd.DataFrame(
        user_computed_data_for_visualization
    ).T

    user_power_profiles_dfs = pd.DataFrame()
    for user_id, user_power_profile in power_profiles.items():
        start_charge_time = df_user_computed_data_for_visualization.loc[
            user_id, "Start charge time"
        ]
        date_index = pd.date_range(
            start=start_charge_time, periods=len(user_power_profile), freq="15min"
        ).ceil("15min")
        power_profiles_df = pd.DataFrame(
            {"date": date_index, "power": user_power_profile}
        )
        power_profiles_df["user_id"] = user_id
        user_power_profiles_dfs = pd.concat(
            [user_power_profiles_dfs, power_profiles_df]
        )

    fig = px.area(
        user_power_profiles_dfs,
        x="date",
        y="power",
        color="user_id",
        title="Simulated Individual Power Profiles",
        labels={"power": "Power (kWh)"},
    )

    fig.add_trace(
        go.Scatter(
            x=aggregate_power_profile["date"],
            y=aggregate_power_profile["power"],
            mode="lines",
            name="Power Profile",
        )
    )

    # Add second y-axis for prices
    fig.update_layout(
        yaxis2=dict(
            title="Prices (cents/kWh)",
            overlaying="y",
            side="right",
        )
    )

    # Add price data to the second y-axis
    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[0] for user_id, price in prices.items()],
            mode="lines",
            name="Scheduled Price",
            yaxis="y2",
            line=dict(dash="dash"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[1] for user_id, price in prices.items()],
            mode="lines",
            name="Regular Price",
            yaxis="y2",
            line=dict(dash="dash"),
        )
    )

    # add a dot at the beginning of each session
    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[0] * len(df_user_computed_data_for_visualization),
            mode="markers",
            marker=dict(
                size=5,
            ),
            showlegend=False,
            hovertemplate=("{text}"),
            text=[
                f"User ID: %{index}<br>"
                f"Start Charge Time: %{row["Start charge time"]}<br>"
                f"Energy needed: %{row["Energy needed"]}<br>"
                f"Duration (hours): %{row["Duration (hours)"]}"
                for index, row in df_user_computed_data_for_visualization.iterrows()
            ],
        )
    )
    fig.update_layout(
        title=f"Simulated Aggregate Power Profile: {min(aggregate_power_profile['date'].dt.date)} to {max(aggregate_power_profile['date'].dt.date)}",
        xaxis_title="Date",
        yaxis_title="Aggregate Station Power (kWh)",
    )
    fig.show()
