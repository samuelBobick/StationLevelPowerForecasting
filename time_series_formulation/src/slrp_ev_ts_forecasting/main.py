import cProfile
import pstats

from slrp_ev_ts_forecasting.default_parameters import (
    TypeDataSet,
    TypeModelChoice,
    TypeScalingMode,
)
from slrp_ev_ts_forecasting.run_one_model import run_one_model

USE_PROFILER = False


def run_one_model_profiled(*args, **kwargs):
    if USE_PROFILER:
        pr = cProfile.Profile()
        pr.enable()

    run_one_model(*args, **kwargs)

    if USE_PROFILER:
        pr.disable()
        with open("profiling_results.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats("cumulative")
            ps.print_stats()


list_model_choices: list[TypeModelChoice] = ["LinearRegression"]
#
number_of_models_per_config = 1
dataset: TypeDataSet = "slrp-ev_new"
session_based_mode = True
peak_prediction = True
list_optimize_lags = [None]  # ["short_opt", None, "long_opt"]
list_scaling_mode: list[TypeScalingMode] = [
    "rolling_standardize",
]

if __name__ == "__main__":
    for model_choice in list_model_choices:
        for get_val_data_from_shuffled_train in [False]:
            for dropout in [0.4]:
                for batch_norm in [True]:
                    for optimize_lags in list_optimize_lags:
                        for scaling_mode in list_scaling_mode:
                            for i in range(number_of_models_per_config):
                                run_one_model_profiled(
                                    model_choice=model_choice,
                                    model_parameters={
                                        "optimize_lags": optimize_lags,
                                        # "dropout": dropout,
                                        # "get_val_data_from_shuffled_train": get_val_data_from_shuffled_train,
                                        # "batch_norm": batch_norm,
                                        "scaling_mode": scaling_mode,
                                        "session_based_mode": session_based_mode,
                                        "peak_prediction": peak_prediction,
                                    },
                                    save_results_filename="improve_peak_forecasting",
                                    dataset=dataset,
                                )
