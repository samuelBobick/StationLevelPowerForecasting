import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd  # Add pandas import
from tqdm.auto import tqdm

from slrp_ev_ts_forecasting.default_parameters import (
    RANDOM_SEED,
    VERBOSE,
    TypeDatasetName,
    TypeModelChoice,
)
from slrp_ev_ts_forecasting.models.dict_models import DICT_MODEL
from slrp_ev_ts_forecasting.run_one_model import run_one_model

# ==================
# Start of USER INPUTS

list_model_choices: list[TypeModelChoice] = [
    # "Basic_NN",
    # "LSTM",
    "TCN",
    # "KNN",
    # "XGBoost",
    # "LinearRegression",
]
dataset: TypeDatasetName = "slrp-ev_new"

search_space = {
    "x_dim": [96 * 2],
    "lookahead": [96],
    # torch models
    "hidden_size": [16, 32, 64, 128],
    "num_hidden_layers": [2, 3, 4],
    "num_lstm_layers": [1, 2, 3],
    "kernel_size": [3, 5, 7],  # TCN
    "max_depth": [4, 6, 8],  # XGBoost
    "epochs": [7],
    "batch_size": [32, 64, 128],
    "batch_norm": [True, False],
    "optimize_lags": [None],  # [None, "long_opt"],
    "dropout": [0, 0.2, 0.4, 0.6],
    "get_val_data_from_shuffled_train": [True],
    "scaling_mode": ["normalize"],
    # [
    #     "normalize",
    #     "standardize",
    #     "rolling_standardize",
    #     "rolling_normalize",
    # ],
    # knn
    "n_neighbors": [4, 7, 10, 13],
    "percentile": [25, 50, 75, 90],
    # parameters for session based forecasting
    "session_based_mode": [False],
    "peak_prediction": [False],
    "add_number_of_sessions": [True, False],
    "add_fraction_of_regular_sessions": [True, False],
    "use_all_active_sessions": [True],
    # parameters to generate random sessions
    "number_of_artificial_datasets": [0],
    "random_start_time": [True, False],
    "shuffle_power_profiles": [True, False],
    "random_power_profile_shapes": [True, False],
    "random_user_needs": [True, False],
    "random_choices": [True, False],
    # parameter for extra features
    "add_number_of_evses_available": [False],
}


number_of_models_per_config = 3
n_random_samples = 100  # Number of random samples to evaluate
parallelize = False
filename_suffix = "initial_models_v2"

# End of USER INPUTS
# ==================


def evaluate_model(model_choice, model_parameters, dataset):
    for i in range(number_of_models_per_config):
        run_one_model(
            model_choice=model_choice,
            model_parameters=model_parameters,
            save_results_filename=f"hyperparameter_search_{model_choice}_{filename_suffix}",
            dataset=dataset,
            verbose=False,
        )


def filter_all_configs(all_configs: list[dict], search_space: dict) -> list[dict]:
    """Filter out some configs that are actually the same"""
    df_all_configs = pd.DataFrame(all_configs)

    df_neutral_values = df_all_configs.apply(
        lambda x: False if False in x.unique() else x.unique()[0]
    )

    # Filter out configs where number_of_artificial_datasets == 0 and random_xx parameters are not their
    # neutral value
    mask = (df_all_configs["number_of_artificial_datasets"] == 0) & (
        (df_all_configs["random_start_time"] != df_neutral_values["random_start_time"])
        | (
            df_all_configs["shuffle_power_profiles"]
            != df_neutral_values["shuffle_power_profiles"]
        )
        | (
            df_all_configs["random_power_profile_shapes"]
            != df_neutral_values["random_power_profile_shapes"]
        )
        | (
            df_all_configs["random_user_needs"]
            != df_neutral_values["random_user_needs"]
        )
        | (df_all_configs["random_choices"] != df_neutral_values["random_choices"])
    )
    df_all_configs = df_all_configs[~mask]

    # Filter out configs where shuffle_power_profiles and random_power_profile_shapes are both True
    df_all_configs = df_all_configs[
        ~(
            df_all_configs["shuffle_power_profiles"]
            & df_all_configs["random_power_profile_shapes"]
        )
    ]

    # Filter out configs where number_of_artificial_datasets is not 0 but all the random_xx parameters are all False
    mask = (df_all_configs["number_of_artificial_datasets"] > 0) & (
        (df_all_configs["random_start_time"] == False)
        & (df_all_configs["shuffle_power_profiles"] == False)
        & (df_all_configs["random_power_profile_shapes"] == False)
        & (df_all_configs["random_user_needs"] == False)
        & (df_all_configs["random_choices"] == False)
    )
    df_all_configs = df_all_configs[~mask]

    # Filter out configs where session_based_mode is False and other session-based parameters are not their
    # neutral value
    if "session_based_mode" in df_all_configs.columns:
        session_based_mask = (df_all_configs["session_based_mode"] == False) & (
            (df_all_configs["peak_prediction"] != False)
            | (
                df_all_configs["add_number_of_sessions"]
                != df_neutral_values["add_number_of_sessions"]
            )
            | (
                df_all_configs["add_fraction_of_regular_sessions"]
                != df_neutral_values["add_fraction_of_regular_sessions"]
            )
            | (
                df_all_configs["use_all_active_sessions"]
                != df_neutral_values["use_all_active_sessions"]
            )
        )
        df_all_configs = df_all_configs[~session_based_mask]

    return df_all_configs.to_dict(orient="records")


def get_random_all_configs_filtered(model_choice):
    model_class = DICT_MODEL[model_choice]["model"]
    model_inputs = model_class.__init__.__code__.co_varnames

    search_space_of_model = {k: v for k, v in search_space.items() if k in model_inputs}
    keys, values = zip(*search_space_of_model.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Initial number of configs: {len(all_configs):,.0f}")

    all_configs = filter_all_configs(all_configs, search_space)
    print(f"Number of configs after filtering: {len(all_configs):,.0f}")

    if len(all_configs) > n_random_samples:
        random_configs = random.sample(all_configs, n_random_samples)
    else:
        random_configs = all_configs

    return random_configs


if __name__ == "__main__":
    assert RANDOM_SEED is None, (
        "Please set RANDOM_SEED to None before running "
        "hyperparameter search to add randomness."
    )
    assert VERBOSE is False, (
        "Please set VERBOSE to False before running "
        "hyperparameter search to avoid printing."
    )
    if parallelize:
        with ThreadPoolExecutor() as executor:
            futures = []
            for model_choice in list_model_choices:
                random_configs = get_random_all_configs_filtered(model_choice)

                for model_parameters in random_configs:
                    futures.append(
                        executor.submit(
                            evaluate_model, model_choice, model_parameters, dataset
                        )
                    )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Hyperparameter Search"
            ):
                future.result()

    else:
        for model_choice in list_model_choices:
            random_configs = get_random_all_configs_filtered(model_choice)

            for model_parameters in tqdm(
                random_configs, desc=f"Search optimal {model_choice}"
            ):
                evaluate_model(model_choice, model_parameters, dataset)
