import os
from multiprocessing import Lock
from typing import Literal, Optional

import pandas as pd
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
from utils.utils_simulation_visualization import (
    visualize_simulation_choices,
    visualize_simulation_power,
    visualize_simulation_prices,
)
from utils.utils_time_and_indexes import (
    get_end_charge_times,
)

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
    if there is a model type. E.g. "peak_forecast_linear" will be parsed to ("peak_forecast", "linear").
    If there is no model type, the second element of the tuple will be None.

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
    data: pd.DataFrame,
    month: int,
    year: int,
    set_choices: Literal["all_scheduled", "all_regular", "standard"],
    date_greater_than: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Helper function to filter the sessions dataframe and keep the relevant ones.

    Args:
        data (pd.DataFrame): dataframe of sessions. This is for example the file `Sessions3.csv`.
        month (int): month to keep, as an integer. 1 = Jan, 2 = Feb, ...
        year (int): year to keep
        set_choices (string): If the set_choices is "all_scheduled", we change the \
            choices of all the sessions to "SCHEDULED". If the set_choices is "all_regular", \
            we change the choice of all sessions to "REGULAR". \
            If the set_choices is "standard", we keep the historical choices.
        date_greater_than (pd.Timestamp, optional): start date to filter the sessions by.

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

    if set_choices == "all_scheduled":
        test_df["choice"] = "SCHEDULED"
    elif set_choices == "all_regular":
        test_df["choice"] = "REGULAR"
    else:
        print(
            "INFO: Using historical choices. You might want to change the set_choices of this function (filter_data) "
            "to 'all_scheduled' or 'all_regular' to change the choices of the user. "
            "WARNING: Historical regular users will have flexibility in their charging requirements ONLY if the "
            "set_choices is set to 'all_scheduled' in this function."
        )

    return test_df


def get_simulator(
    data: pd.DataFrame,
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
    """Get the simulation object based on the scenario name."""
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
    sim: BaselineSimulator,
    results_file_name: str,
    summary_file_name: str,
    aggregate_power_profile_file_name: str,
    verbose=False,
    visualize=False,
):
    """Simulate the given sim.
    Save the simulation in results_file_name and append to summary file located at summary_file_name.
    Also print summary information and plots of the simulation if verbose is True.

    Args:
        sim (BaselineSimulator): simulator
        results_file_name (string): filepath to store result dataframe
        summary_file_name (string): filepath of summary dataframe. If summary dataframe doesn't exist, creates a .csv file here.
        aggregate_power_profile_file_name (string): filepath of aggregate_power_profile dataframe. If summary dataframe doesn't exist, creates a .csv file here.
        verbose (bool, optional): If true, prints summary information of the results. Defaults to False.
        visualize (bool, optional): If true, visualizes the results. Defaults to False.
    """
    # parse the months from the simulation data, and make sure we only have 1 month
    months = pd.to_datetime(sim.test_df["startChargeTime"]).dt.month.unique()
    if len(months) > 1:
        raise ValueError(
            "You need to run the simulation on a single month."
            "Please make sure the simulation data only contains 1 month"
        )
    month = months[0]

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

        visualize_simulation_prices(
            session_results,
            TOU=sim.TOU,
            aggregate_power_profile=sim.aggregate_power_profile,
        )

        visualize_simulation_choices(session_results)


def compute_prediction_error(
    aggregate_power_profile, user_computed_data_for_visualization: dict
) -> tuple[Losses, Losses, Losses | None]:
    """
    Computes error of peak predictions for each optimization.
    This function is run after the simulation for logging in summary.csv.

    Args:
        aggregate_power_profile (pd.DataFrame): DataFrame with the columns 'date' and 'power' containing the aggregate load at the station.
        user_computed_data_for_visualization (pd.DataFrame): Cached data from simulation.

    Raises:
        ValueError: Error is thrown if the simulation length is less than one full day

    Returns:
        tuple[Losses, Losses, Losses | None]:
            (peak prediction | m = SCHEDULED,
            peak prediction | m = REGULAR,
            initial peak foreacast: assumes m = REGULAR and existing SCHEDULED sessions cannot be reoptimized)
    """
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
            "You need to have more than 1 day of data to compute the RMSE of the peak"
            "predictions. Please rerun the simulation with more days."
        )

    sch_losses = _compute_one_prediction_error(
        "Peak pred (sch)",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    reg_losses = _compute_one_prediction_error(
        "Peak pred (reg)",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    initial_pred_losses = _compute_one_prediction_error(
        "Peak initial forecast",
        df_user_computed_data_for_visualization,
        real_values,
        last_day,
    )

    return sch_losses, reg_losses, initial_pred_losses


def _compute_one_prediction_error(
    prediction_column: str,
    df_user_computed_data_for_visualization: pd.DataFrame,
    real_values,
    last_day: pd.Timestamp,
) -> Losses:
    """Computes one type of prediction error

    Args:
        prediction_column (str): either "Peak pred (sch)", "Peak pred (reg)", or "Peak initial foreacst"
        df_user_computed_data_for_visualization (pd.DataFrame): cached data from simulation
        real_values (pd.Series): ground truth peak values
        last_day (pd.Timestamp): last day of the simulation

    Returns:
        Losses: Losses object (see time_series_formulation/src/slrp_ev_ts_forecasting/compute_losses.py)
    """

    these_peak_predictions = df_user_computed_data_for_visualization[prediction_column]
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
