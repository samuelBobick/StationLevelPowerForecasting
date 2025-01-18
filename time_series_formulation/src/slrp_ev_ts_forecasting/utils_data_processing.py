import pandas as pd
from slrp_ev_data.feature_engineering import (
    convert_date_from_datetime_to_int,
    reverse_feature_engineering,
)
from slrp_ev_data.normalization_and_standardization import (
    get_rolling_scaling_column,
    get_train_mean_and_std,
    get_train_min_and_max,
    retrieve_train_mean_and_std,
    retrieve_train_min_and_max,
)

from slrp_ev_ts_forecasting.default_parameters import TypeDataSet, TypeScalingMode


def reverse_engineer_forecast(
    df_test_example,
    df_predictions,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
) -> pd.DataFrame:
    # Reverse engineer the forecast to get the original features back
    # initialize final dataframe
    df_reversed_predictions = pd.DataFrame()
    df_reversed_predictions["date"] = df_predictions["date"]

    # convert from float32 to int64
    df_predictions["date"] = convert_date_from_datetime_to_int(df_predictions["date"])

    list_columns = list(df_predictions.columns)
    for i, col_name in enumerate(list_columns):
        if col_name == "date":
            continue

        # merge_asof performs a left merge with the closest date
        df_reverse_helper = pd.merge_asof(
            df_predictions[["date", col_name]],
            df_test_example.drop(columns=["power"]),
            on="date",
            direction="nearest",
        ).rename(columns={col_name: "power"})
        df_reverse_helper = df_reverse_helper.dropna(subset=["power"])
        df_reverse_helper = reverse_feature_engineering(
            df_reverse_helper,
            scaling_mode,
            scaling_parameters,
            bypass_output_validation=True,
        )

        df_reverse_helper = df_reverse_helper[["date", "power"]]
        helper_date_mask = df_reversed_predictions["date"].isin(
            df_reverse_helper["date"]
        )
        df_reversed_predictions.loc[helper_date_mask, col_name] = df_reverse_helper[
            "power"
        ]

    return df_reversed_predictions


def get_scaling_parameters(
    train: pd.DataFrame | None,
    data: pd.DataFrame | None,
    data_scaling_mode: TypeScalingMode,
    dataset: TypeDataSet,
    lookahead_15min_steps: int,
    retrieve_from_saved: bool = False,
) -> tuple | pd.DataFrame:
    if data_scaling_mode == "normalize":
        if retrieve_from_saved:
            scaling_parameters = retrieve_train_min_and_max(dataset_name=dataset)
        else:
            if train is None:
                raise ValueError(
                    "train dataframe must be provided when data_scaling_mode is 'normalize'"
                )
            scaling_parameters = get_train_min_and_max(train, dataset_name=dataset)

    elif data_scaling_mode == "standardize":
        if retrieve_from_saved:
            scaling_parameters = retrieve_train_mean_and_std(dataset_name=dataset)
        else:
            if train is None:
                raise ValueError(
                    "train dataframe must be provided when data_scaling_mode is 'standardize'"
                )
            scaling_parameters = get_train_mean_and_std(train, dataset_name=dataset)

    elif data_scaling_mode in ["rolling_standardize", "rolling_normalize"]:
        if train is None or data is None:
            raise ValueError(
                "train and data dataframes must be provided when data_scaling_mode is 'rolling_standardize'"
            )

        scaling_parameters = get_rolling_scaling_column(
            data,
            scaling_mode=data_scaling_mode,
            lookahead_15min_steps=lookahead_15min_steps,
            dataset_name=dataset,
        )

    else:
        raise ValueError(
            f"Data scaling mode {data_scaling_mode} is not defined. Please refer to "
            "TypeDataScalingMode for supported data scaling modes."
        )
    return scaling_parameters
