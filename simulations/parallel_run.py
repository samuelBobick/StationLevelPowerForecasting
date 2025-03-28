import concurrent.futures
import datetime
import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from utils.simulation_utils import filter_data, generate_session_results, get_simulator


# Move this function to the top level so it can be pickled
def run_simulation(
    i, month, year, scenario, sessions_df, folder_path, eps, max_retries=5
):
    run_verbose = False
    final_verbose = False
    filter_date_greater_than: pd.Timestamp | None = None

    initial_running_peak: float = 0
    results_file_name = f"{folder_path}/{month}_{year}_{scenario}_{i}.csv"
    aggregate_power_profile_file_name = f"{folder_path}/aggregate_power_profile.csv"
    summary_file_name = f"{folder_path}/summary.csv"
    retry_count = 0
    success = False

    while not success and retry_count < max_retries:
        try:
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
            sim.eps = eps

            generate_session_results(
                sim,
                month,
                results_file_name,
                summary_file_name,
                aggregate_power_profile_file_name,
                visualize=final_verbose,
                verbose=final_verbose,
            )
            success = True
        except Exception as e:
            print(f"An error occurred: {e}")
            eps += 0.02
            retry_count += 1
            print(
                f"Retrying with eps = {eps} (attempt {retry_count + 1}/{max_retries})"
            )

    if not success:
        print(f"Maximum retry attempts reached for iteration {i} in month {month}")


def main():
    months = range(8, 10)
    year = 2023
    num_iterations = 20

    for scenario in [
        "standard",
        "timeseries_forecast_naive",
        "timeseries_forecast_linear",
        "timeseries_forecast_xgboost",
        "smooth_dc_penalty",
        "threshold",
    ]:
        sessions_file = Path(__file__).resolve().parents[1] / "data" / "Sessions3.csv"
        sessions_df = pd.read_csv(sessions_file).sort_values(by="startChargeTime")

        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = f"results/{scenario}/{time_str}"
        os.makedirs(folder_path, exist_ok=True)

        eps = 0.02

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
                    eps,
                )
                for i in range(num_iterations)
                for month in months
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
