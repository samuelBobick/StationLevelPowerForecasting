import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from slrp_ev_ts_forecasting.default_parameters import (
    RANDOM_SEED,
    VERBOSE,
    TypeDataSet,
    TypeModelChoice,
)
from slrp_ev_ts_forecasting.models.dict_models import DICT_MODEL
from slrp_ev_ts_forecasting.run_one_model import run_one_model

# ==================
# Start of USER INPUTS

list_model_choices: list[TypeModelChoice] = ["XGBoost"]
dataset: TypeDataSet = "slrp-ev_new"

search_space = {
    "x_dim": [96 * 2],
    "lookahead": [96],
    # torch models
    "hidden_size": [64, 128, 256],
    "num_hidden_layers": [2, 3, 4],
    "num_lstm_layers": [1, 2, 3],
    "kernel_size": [3, 5, 7],  # TCN
    "max_depth": [4],  # [4, 6, 8],  # XGBoost
    "epochs": [5, 10],
    "batch_size": [32, 64, 128],
    "batch_norm": [True, False],
    "optimize_lags": [None],  # [None, "long_opt"],
    "dropout": [0, 0.2, 0.4, 0.6],
    "get_val_data_from_shuffled_train": [False],
    "scaling_mode": [
        "normalize",
        "standardize",
        "rolling_standardize",
        "rolling_normalize",
    ],
    "session_based_mode": [False],
    "peak_prediction": [False],
    "add_number_of_sessions": [True, False],
    "add_fraction_of_regular_sessions": [True, False],
    "use_all_active_sessions": [True],
    "number_of_artificial_datasets": [0],
    "random_start_time": [True, False],
    "shuffle_power_profiles": [True, False],
    "random_power_profile_shapes": [True, False],
    "random_user_needs": [True, False],
    "random_choices": [True, False],
    "add_number_of_evses_available": [True, False],
}


number_of_models_per_config = 3
n_random_samples = 100  # Number of random samples to evaluate
parallelize = False

# End of USER INPUTS
# ==================


def evaluate_model(model_choice, model_parameters, dataset):
    for i in range(number_of_models_per_config):
        run_one_model(
            model_choice=model_choice,
            model_parameters=model_parameters,
            save_results_filename=f"hyperparameter_search_{model_choice}_dropout",
            dataset=dataset,
            verbose=False,
        )


def filter_all_configs(all_configs, search_space):
    """Filter out some configs that are actually the same"""
    # filter out some configs that are actually the same
    # if number_of_artificial_datasets == 0, then all config where "random_xx" parameters are not False should be removed
    config_index_to_remove = []
    for i in range(len(all_configs)):
        config = all_configs[i]
        if config["number_of_artificial_datasets"] == 0:
            artificial_dataset_config = {
                "random_start_time": config["random_start_time"],
                "shuffle_power_profiles": config["shuffle_power_profiles"],
                "random_power_profile_shapes": config["random_power_profile_shapes"],
                "random_user_needs": config["random_user_needs"],
                "random_choices": config["random_choices"],
            }
            if not _check_first_config_parameters(
                artificial_dataset_config, search_space
            ):
                config_index_to_remove.append(i)
        if config["shuffle_power_profiles"] and config["random_power_profile_shapes"]:
            config_index_to_remove.append(i)
        if not config["session_based_mode"]:
            session_based_config = {
                "peak_prediction": config["peak_prediction"],
                "add_number_of_sessions": config["add_number_of_sessions"],
                "add_fraction_of_regular_sessions": config[
                    "add_fraction_of_regular_sessions"
                ],
                "use_all_active_sessions": config["use_all_active_sessions"],
            }
            if not _check_first_config_parameters(session_based_config, search_space):
                config_index_to_remove.append(i)

    # remove the configs
    all_configs = [
        all_configs[i]
        for i in range(len(all_configs))
        if i not in config_index_to_remove
    ]
    return all_configs


def _check_first_config_parameters(config: dict, search_space: dict):
    """Check that all the parameters have their first possible value.
    this function is used to remove some of the configs, in case the
    parameters are irrelevant."""
    for k, v in config.items():
        if False in search_space[k]:
            if v is not False:
                return False
        elif v != search_space[k][0]:
            return False
    return True


def get_random_all_configs_filtered(model_choice):
    model_class = DICT_MODEL[model_choice]["model"]
    model_inputs = model_class.__init__.__code__.co_varnames

    search_space_of_model = {k: v for k, v in search_space.items() if k in model_inputs}
    keys, values = zip(*search_space_of_model.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_configs = filter_all_configs(all_configs, search_space)

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
