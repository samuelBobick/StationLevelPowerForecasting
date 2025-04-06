import concurrent.futures
import datetime
import os
import warnings
from pathlib import Path

import pandas as pd
from utils.simulation_utils import (
    TypeScenario,
    filter_data,
    generate_session_results,
    get_simulator,
)

warnings.filterwarnings("ignore")


def run_simulation(i, month, year, scenario, sessions_df, folder_path):
    """
    Run simulations in parallel.

    Args:
        i (int): current iteration number.
        month (int): 1-12, representing the month to simulate.
        year (int): year to simulate.
        scenario (TypeScenario): type of simulation to run.
        sessions_df (pd.DataFrame): dataframe of sessions in the format of sessions3.csv.
        folder_path (String): path to save the simulation results.
    """
    run_verbose = False
    final_verbose = False
    filter_date_greater_than: pd.Timestamp | None = None
    initial_running_peak: float = 0

    results_file_name = f"{folder_path}/{month}_{year}_{scenario}_{i}.csv"
    aggregate_power_profile_file_name = (
        f"{folder_path}/aggregate_power_profile_{month}_{year}_{i}.csv"
    )
    summary_file_name = f"{folder_path}/summary.csv"

    test_df = filter_data(
        sessions_df, month, year, "all_scheduled", filter_date_greater_than
    )
    sim = get_simulator(
        test_df,
        scenario,
        verbose=run_verbose,
        initial_running_peak=initial_running_peak,
        monte_carlo=True,
    )

    generate_session_results(
        sim,
        month,
        results_file_name,
        summary_file_name,
        aggregate_power_profile_file_name,
        visualize=final_verbose,
        verbose=final_verbose,
    )


def main(
    scenarios: list[TypeScenario],
    months: range | list[int],
    year: int,
    num_iterations_of_each_month: int,
):
    """
    Simulate months in parallel

    Args:
        scenarios (list[TypeScenario]): list of scenarios to simulate.
        months (range | list[int]): months to simulate.
        year (int): year to simulate.
        num_iterations_of_each_month (int): number of simulations to run per month.
    """
    for scenario in scenarios:
        sessions_file = Path(__file__).resolve().parents[1] / "data" / "Sessions3.csv"
        sessions_df = pd.read_csv(sessions_file).sort_values(by="startChargeTime")

        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = f"results/{scenario}/{time_str}"
        os.makedirs(folder_path, exist_ok=True)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = {
                executor.submit(
                    run_simulation,
                    i,
                    month,
                    year,
                    scenario,
                    sessions_df,
                    folder_path,
                )
                for i in range(num_iterations_of_each_month)
                for month in months
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main(
        scenarios=[
            "standard",
            "smooth_dc_penalty",
            "timeseries_forecast_naive",
            "timeseries_forecast_linear",
            "timeseries_forecast_xgboost",
            "threshold",
        ],
        months=range(1, 13),
        year=2023,
        num_iterations_of_each_month=10,
    )
