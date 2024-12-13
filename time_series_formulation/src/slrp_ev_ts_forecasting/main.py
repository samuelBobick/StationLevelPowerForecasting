from slrp_ev_ts_forecasting.default_parameters import TypeDataSet, TypeModelChoice
from slrp_ev_ts_forecasting.run_one_model import run_one_model

model_choices: list[TypeModelChoice] = ["LinearRegression"]
number_of_models_per_config = 1
dataset: TypeDataSet = "slrp-ev_new"
session_based_mode = True
peak_prediction = True

if __name__ == "__main__":
    for model_choice in model_choices:
        for get_val_data_from_shuffled_train in [False]:
            for dropout in [0.4]:
                for batch_norm in [True]:
                    for i in range(number_of_models_per_config):
                        run_one_model(
                            model_choice=model_choice,
                            model_parameters={
                                # "dropout": dropout,
                                # "get_val_data_from_shuffled_train": get_val_data_from_shuffled_train,
                                # "batch_norm": batch_norm,
                                "session_based_mode": session_based_mode,
                                "peak_prediction": peak_prediction,
                            },
                            save_results_filename="results_linear_model",
                            dataset=dataset,
                        )
