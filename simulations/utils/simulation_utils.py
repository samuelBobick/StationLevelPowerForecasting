import os
from multiprocessing import Lock
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from baseline_simulator import BaselineSimulator
from constants.tariffs import MODIFIED_DC, TypeTariffName
from peak_forecast_simulator import PeakForecastSimulator
from slrp_ev_ts_forecasting.compute_losses import Losses, compute_losses
from slrp_ev_ts_forecasting.save_losses import print_losses
from smooth_dc_penalty_simulator import SmoothDCPenaltySimulator
from threshold_simulator import ThresholdSimulator
from timeseries_forecast_simulator import TimeseriesForecastSimulator
from utils.utils import (
    get_profit,
    get_session_results,
)
from utils.utils_time_and_indexes import (
    convert_power_profile_to_df,
    convert_time_to_index,
    get_end_charge_times,
)
from utils.utils_visualization import create_tou_heatmap_trace

TypeScenario = Literal[
    "all_scheduled",
    "all_regular",
    "standard",
    "smooth_dc_penalty",
    "threshold",
    "peak_forecast_linear",
    "timeseries_forecast_naive",
    "timeseries_forecast_linear",
    "timeseries_forecast_xgboost",
]


def parse_type_scenario(scenario: TypeScenario) -> tuple[str, str | None]:
    """Parse the scenario string to separate the scenario name and the model_type,
    if there is a model type

    Args:
        scenario (TypeScenario): scenario name
    """
    if "peak_forecast" in scenario:
        return "peak_forecast", scenario.split("_")[-1]
    elif "timeseries_forecast" in scenario:
        return "timeseries_forecast", scenario.split("_")[-1]
    else:
        return scenario, None


def filter_data(
    data,
    month,
    year,
    scenario: TypeScenario,
    date_greater_than: Optional[pd.Timestamp] = None,
):
    """Helper function to filter the sessions dataframe

    Args:
        data (pd.DataFrame): dataframe of sessions
        month (int): month, as an integer. 1 = Jan, 2 = Feb, ...
        year (int): year
        scenario (string): if the scenario

    Returns:
        test_df (pd.DataFrame): filtered dataframe
    """
    date_connect_time = pd.to_datetime(data["connectTime"])

    # handle case where date_greater_than is None
    if date_greater_than is None:
        date_greater_than = date_connect_time.min()
    else:
        print(f"INFO: using sessions starting AFTER {date_greater_than}")

    # filter based on month and year inputs
    test_df = data[
        (date_connect_time.dt.year == year)
        & (date_connect_time.dt.month == month)
        & (date_connect_time > date_greater_than)
    ]

    # Keep meaningful sessions
    test_df = test_df[test_df["DurationHrs"] > 0.5]
    test_df = test_df[test_df["cumEnergy_Wh"] > 0]

    # Drop overnight sessions (so when the end charge time is the next day)
    end_charge_time = get_end_charge_times(test_df)
    start_charge_time = pd.to_datetime(test_df["connectTime"])
    print(
        f"INFO: Dropping {len(test_df) - len(test_df[(end_charge_time.dt.day - start_charge_time.dt.day) == 0])} overnight sessions"
    )
    test_df = test_df[(end_charge_time.dt.day - start_charge_time.dt.day) == 0]

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
    initial_running_peak: float = 0,
):
    """_summary_

    Args:
        scenario (TypeScenario): _description_
    """
    if initial_running_peak > 0:
        print(f"INFO: Using initial running peak of {initial_running_peak} kW")
        print("---------------- Starting simulation ----------------")

    scenario_name, model_type = parse_type_scenario(scenario)

    if scenario_name in ["all_scheduled", "all_regular", "standard"]:
        return BaselineSimulator(
            data,
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
    elif scenario_name == "smooth_dc_penalty":
        return SmoothDCPenaltySimulator(
            data,
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
    elif scenario_name == "threshold":
        return ThresholdSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            initial_running_peak,
            monte_carlo,
            verbose,
            step,
        )
    elif scenario_name == "peak_forecast":
        if model_type is None:
            raise ValueError(
                "Model type not provided for peak_forecast scenario. \
                The model type should be added in the scenario name as {mpc_scenario}_{model_type}. \
                Please refer to TypeScenario."
            )
        return PeakForecastSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            initial_running_peak,
            monte_carlo,
            verbose,
            model_type=model_type,  # type: ignore
        )
    elif scenario_name == "timeseries_forecast":
        if model_type is None:
            raise ValueError(
                "Model type not provided for timeseries_forecast scenario. \
                The model type should be added in the scenario name as {mpc_scenario}_{model_type}. \
                Please refer to TypeScenario."
            )
        return TimeseriesForecastSimulator(
            data,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            initial_running_peak,
            monte_carlo,
            verbose,
            model_type=model_type,  # type: ignore
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

    charging_revenue_cents, TOU_cost_cents = get_profit(session_results)

    demand_charge_kw = round(max(sim.aggregate_power_profile["power"]), 2)
    demand_charge_cents = sim.cost_dc * demand_charge_kw
    total_profit = round(
        (charging_revenue_cents - TOU_cost_cents - demand_charge_cents) / 100, 2
    )
    charging_revenue = round(charging_revenue_cents / 100, 2)
    TOU_cost = round(TOU_cost_cents / 100, 2)
    demand_charge = round(demand_charge_cents / 100, 2)
    energy_delivered = round(sum(session_results["energy_delivered"]), 2)

    sch_losses, reg_losses, initial_pred_losses = compute_prediction_error(
        sim.aggregate_power_profile, user_computed_data_for_visualization
    )

    sch_rmse = round(sch_losses["rmse"], 2)
    reg_rmse = round(reg_losses["rmse"], 2)
    initial_pred_rmse = (
        round(initial_pred_losses["rmse"], 2) if initial_pred_losses else None
    )

    print(
        "Profit", total_profit, "\nTOU Cost", TOU_cost, "\nDemand Charge", demand_charge
    )

    row_data = {
        "Month": month,
        "Total Profit ($)": total_profit,
        "Charging Revenue ($)": charging_revenue,
        "TOU Cost ($)": TOU_cost,
        "Demand Charge ($)": demand_charge,
        "Peak Power (kW)": demand_charge_kw,
        "Energy Delivered (kWh)": energy_delivered,
        "Scheduled RMSE": sch_rmse,
        "Regular RMSE": reg_rmse,
        "Initial Prediction RMSE": initial_pred_rmse,
    }

    if os.path.exists(summary_file_name):
        summary_df = pd.read_csv(summary_file_name)
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([row_data])],
            ignore_index=True,
        )
    else:
        columns = [
            "Month",
            "Total Profit ($)",
            "Charging Revenue ($)",
            "TOU Cost ($)",
            "Demand Charge ($)",
            "Peak Power (kW)",
            "Energy Delivered (kWh)",
            "Scheduled RMSE",
            "Regular RMSE",
            "Initial Prediction RMSE",
        ]

        summary_df = pd.DataFrame([row_data], columns=columns)

    sim.aggregate_power_profile.to_csv(aggregate_power_profile_file_name, index=False)
    lock = Lock()

    with lock:
        summary_df.to_csv(summary_file_name, index=False)

    if verbose:
        print("---------------- Summary results ----------------")
        for key in row_data:
            print(f"{key}: {row_data[key]}")
    if visualize:
        print_losses(sch_losses, "Scheduled Peak Prediction: ")
        print_losses(reg_losses, "Regular Peak Prediction: ")
        if initial_pred_losses:
            print_losses(
                initial_pred_losses, "Initial Peak Prediction (timeseries forecast): "
            )

        visualize_simulation_power(
            sim.aggregate_power_profile,
            power_profiles,
            user_computed_data_for_visualization,
            prices,
            sim.delta_t,
            sim.TOU,
        )

        visualize_simulation_prices(session_results, TOU=sim.TOU)

        visualize_simulation_choices(session_results)


def compute_prediction_error(
    aggregate_power_profile, user_computed_data_for_visualization: pd.DataFrame
) -> tuple[Losses, Losses, Losses | None]:
    # turn dictionaries into dataframe for easier manipulation
    df_user_computed_data_for_visualization = pd.DataFrame(
        user_computed_data_for_visualization
    ).T.set_index("Start charge time")

    real_values = df_user_computed_data_for_visualization.apply(
        _apply_get_real_peak_value,
        axis=1,
        aggregate_power_profile=aggregate_power_profile,
    )

    # filter out the values for which the day is not over yet
    last_day = real_values.index[-1].date()
    real_values = real_values.loc[real_values.index.date < real_values.index.date.max()]
    if real_values.empty:
        raise ValueError(
            "You need to have more than 1 day of data to compute the RMSE of the peak \
            predictions. Please rerun the simulation with more days."
        )

    sch_losses = _compute_one_type_of_prediction_losses(
        "Peak pred (sch)",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    reg_losses = _compute_one_type_of_prediction_losses(
        "Peak pred (reg)",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    initial_pred_losses = _compute_one_type_of_prediction_losses(
        "Peak initial forecast",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    return sch_losses, reg_losses, initial_pred_losses


def _compute_one_type_of_prediction_losses(
    prediction_key: str,
    df_user_computed_data_for_visualization: pd.DataFrame,
    real_values,
    last_day: pd.Timestamp,
) -> Losses:

    these_peak_predictions = df_user_computed_data_for_visualization[prediction_key]
    if pd.isna(these_peak_predictions).any():
        return None  # type: ignore

    these_peak_predictions = these_peak_predictions.loc[
        these_peak_predictions.index.date < last_day
    ]

    these_losses = compute_losses(
        these_peak_predictions.values, real_values.values, alpha=2
    )

    return these_losses


def _apply_get_real_peak_value(
    row,
    aggregate_power_profile,
    mode: Literal["peak_of_day", "peak_next_8h"] = "peak_next_8h",
):
    if mode == "peak_of_day":
        mask = aggregate_power_profile["date"].dt.date == row.name.date()
    elif mode == "peak_next_8h":
        mask = (aggregate_power_profile["date"] >= row.name) & (
            aggregate_power_profile["date"] <= row.name + pd.Timedelta(hours=8)
        )
    else:
        raise ValueError("Invalid peak prediction mode")

    return aggregate_power_profile[mask]["power"].max()


def visualize_simulation_power(
    aggregate_power_profile: pd.DataFrame,
    power_profiles: dict,
    user_computed_data_for_visualization: dict,
    prices: dict,
    delta_t: float,
    TOU: np.ndarray,
) -> None:
    """
    Visualize the results of the simulation values. the x axis is time

    Args:
        aggregate_power_profile: dataframe with columns "date" and "power"
    """
    # turn dictionaries into dataframe for easier manipulation
    df_user_computed_data_for_visualization = pd.DataFrame(
        user_computed_data_for_visualization
    ).T

    user_power_profiles_dfs = pd.DataFrame()
    for user_id, user_power_profile in power_profiles.items():
        start_charge_time = df_user_computed_data_for_visualization.loc[
            user_id, "Start charge time"
        ]
        power_profiles_df = convert_power_profile_to_df(
            user_power_profile, start_charge_time, delta_t=delta_t
        )

        power_profiles_df["user_id"] = user_id
        user_power_profiles_dfs = pd.concat(
            [user_power_profiles_dfs, power_profiles_df]
        )

        df_user_computed_data_for_visualization.loc[user_id, "Energy delivered"] = (
            np.round(sum(power_profiles_df["power"] * delta_t), 2)
        )

    fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, 60])))

    # Change the background color based on the TOU prices (as bars)
    TOU_data_index = aggregate_power_profile["date"]
    # Add a heatmap for TOU values as background
    fig.add_trace(create_tou_heatmap_trace(TOU_data_index, TOU, unit="kW"))

    # Create a color map for user IDs
    user_ids = user_power_profiles_dfs["user_id"].unique()
    color_scale = plotly.colors.qualitative.Pastel1
    color_scale_darker = (
        plotly.colors.qualitative.Set1
    )  # same colors as pastel palette but darker
    length_color_scale = len(color_scale)
    colors = {
        user_id: color_scale[i % length_color_scale]
        for i, user_id in enumerate(user_ids)
    }

    # Create stacked area plots for each user
    for i, (user_id, user_power_profile_df) in enumerate(
        user_power_profiles_dfs.groupby("user_id")
    ):
        fig.add_trace(
            go.Scatter(
                x=user_power_profile_df["date"],
                y=user_power_profile_df["power"],
                mode="none",
                name=f"User {user_id}",
                line_shape="hv",
                # this parameter will make the line go horizontally first and then vertically (steps)
                stackgroup="one",
                fillcolor=colors[user_id],
            )
        )

    # Aggregate power profile
    fig.add_trace(
        go.Scatter(
            x=aggregate_power_profile["date"],
            y=aggregate_power_profile["power"],
            mode="lines",
            name="Power Profile",
            line=dict(color=color_scale_darker[1], shape="hv"),  # blue
        )
    )

    # Add price data to the second y-axis
    fig.update_layout(
        yaxis2=dict(
            title="Prices (cents/kWh)", overlaying="y", side="right", range=[0, None]
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[0] for user_id, price in prices.items()],
            mode="lines+markers",
            name="Scheduled Price",
            yaxis="y2",
            line=dict(dash="dash", color=color_scale_darker[2]),  # green
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[1] for user_id, price in prices.items()],
            mode="lines+markers",
            name="Regular Price",
            yaxis="y2",
            line=dict(dash="dash", color=color_scale_darker[0]),  # red
        )
    )

    # add a dot at the beginning of each session
    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[0] * len(df_user_computed_data_for_visualization),
            mode="markers",
            marker=dict(
                size=10,
                color=[
                    colors[user_id]
                    for user_id in df_user_computed_data_for_visualization.index
                ],
            ),
            showlegend=False,
            hovertemplate=(
                "User ID: %{customdata[0]}<br>"
                "Start Charge Time: %{x}<br>"
                "Energy needed: %{customdata[1]}<br>"
                "Energy delivered: %{customdata[2]}<br>"
                "Duration (hours): %{customdata[3]}<br>"
                "Choice: %{customdata[4]}<extra></extra>"
            ),
            customdata=np.array(
                [
                    df_user_computed_data_for_visualization.index,
                    df_user_computed_data_for_visualization["Energy needed"],
                    df_user_computed_data_for_visualization["Energy delivered"],
                    df_user_computed_data_for_visualization["Duration (hours)"],
                    df_user_computed_data_for_visualization["Choice"],
                ]
            ).T,
        )
    )

    fig.update_layout(
        title=f"Simulated Aggregate Power Profile: {min(aggregate_power_profile['date'].dt.date)} to {max(aggregate_power_profile['date'].dt.date)}",
        xaxis_title="Date",
        yaxis_title="Aggregate Station Power (kWh)",
    )
    fig.show()


def visualize_simulation_prices(
    session_results: pd.DataFrame,
    box_plot: bool = False,
    TOU: Optional[np.ndarray] = None,
):
    """Creates a figure with a boxplot (or just scatter) of the prices, grouped by hours.
    On the x axis, we have hours, while on the y axis we have prices.
    For each hour, there are 2 boxes, one for the scheduled prices, and one for the regular prices.
    The prices are in cents/kWh.

    Args:
        session_results (pd.DataFrame): output DataFrame of the function get_session_results
        box_plot (bool): if True, creates a box plot. If False, creates a scatter plot.
        TOU (np.ndarray, optional): Time-of-Use price array for adding background heatmap. Defaults to None.
    """
    df_session_prices = session_results[["start_time", "z_sch", "z_reg"]].copy()
    df_session_prices["hour"] = df_session_prices["start_time"].dt.hour
    df_session_prices = df_session_prices.drop(columns="start_time")

    fig = go.Figure()

    # Add TOU heatmap in the background if TOU is provided
    if TOU is not None:
        whole_time_index = pd.Series(
            data=np.arange(
                start=df_session_prices["hour"].min(),
                stop=df_session_prices["hour"].max(),
                step=0.25,
            )
        )
        TOU_current_idx = convert_time_to_index(
            pd.to_datetime("2020-01-01 00:00:00")
            + pd.Timedelta(hours=df_session_prices["hour"].min()),
            0.25,
        )
        fig.add_trace(
            create_tou_heatmap_trace(
                whole_time_index, TOU, TOU_current_idx=TOU_current_idx, unit="kW"
            )
        )

    for price_type in ["z_sch", "z_reg"]:
        color = "green" if price_type == "z_sch" else "red"
        if box_plot:
            fig.add_trace(
                go.Box(
                    x=df_session_prices["hour"],
                    y=df_session_prices[price_type],
                    name=price_type,
                    boxmean="sd",
                    marker=dict(color=color),
                )
            )
        else:
            df_average_prices = df_session_prices.groupby("hour")[price_type].mean()
            fig.add_trace(
                go.Scatter(
                    x=df_average_prices.index,
                    y=df_average_prices,
                    mode="lines+markers",
                    name=price_type,
                    marker=dict(color=color),
                    line=dict(dash="dash"),
                )
            )

    title = "Distribution of average prices per hour"
    # add average regular and scheduled prices in the title
    title += f"<br>    Average scheduled price: {df_session_prices['z_sch'].mean():.1f} cents/kWh"
    title += f"<br>    Average regular price: {df_session_prices['z_reg'].mean():.1f} cents/kWh"

    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title="Average price (cents/kWh)",
        template="plotly_white",
    )

    fig.show()


def visualize_simulation_choices(session_results: pd.DataFrame):
    """Creates a figure with a bar plot. Each hour has one bar, representing the
    fraction of people that chose scheduled charging.

    Args:
        session_results (pd.DataFrame): output DataFrame of the function get_session_results
    """

    df_session_choices = session_results[["start_time", "choice"]].copy()
    df_session_choices["hour"] = df_session_choices["start_time"].dt.hour
    df_session_choices = df_session_choices.drop(columns="start_time")
    df_session_choices = (
        df_session_choices.groupby(["hour"]).value_counts(normalize=True).unstack()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_session_choices.index,
            y=df_session_choices["SCHEDULED"],
            name="choice",
            marker=dict(color="blue"),
        )
    )

    title = "Choice of charging per hour"
    # add average ratio of people choosing scheduled charging in the title, and total number of people
    title += f"<br>    Average fraction of users choosing scheduled charging: {df_session_choices['SCHEDULED'].mean():.2f}"
    title += f"<br>    Total number of users: {session_results.shape[0]}"

    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title="Fraction of users choosing scheduled charging",
        template="plotly_white",
    )

    fig.show()
