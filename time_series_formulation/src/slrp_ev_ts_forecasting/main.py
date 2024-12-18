from slrp_ev_ts_forecasting.default_parameters import TypeDataSet, TypeModelChoice
from slrp_ev_ts_forecasting.run_one_model import run_one_model

list_model_choices: list[TypeModelChoice] = ["PeakPersistence"]
number_of_models_per_config = 1
dataset: TypeDataSet = "slrp-ev_new"
session_based_mode = True
peak_prediction = True
list_optimize_lags = [None]  # ["short_opt", None, "long_opt"]

if __name__ == "__main__":
    for model_choice in list_model_choices:
        for get_val_data_from_shuffled_train in [False]:
            for dropout in [0.4]:
                for batch_norm in [True]:
                    for optimize_lags in list_optimize_lags:
                        for i in range(number_of_models_per_config):
                            run_one_model(
                                model_choice=model_choice,
                                model_parameters={
                                    "optimize_lags": optimize_lags,
                                    # "dropout": dropout,
                                    # "get_val_data_from_shuffled_train": get_val_data_from_shuffled_train,
                                    # "batch_norm": batch_norm,
                                    "session_based_mode": session_based_mode,
                                    "peak_prediction": peak_prediction,
                                },
                                save_results_filename="results_linear_model",
                                dataset=dataset,
                            )
