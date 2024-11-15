from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from slrp_ev_ts_forecasting.default_parameters import (
    DEFAULT_RESULTS_FILENAME,
    RESULTS_PATH,
)

TypeMetrics = Literal["rmse", "wrmse (alpha=2)", "mae", "wprmse (beta=3)", "r2"]
GROUPBY_COLUMNS = [
    "batch_size",
    "x_dim",
    "lookahead",
    "time_mode",
    "dataset",
    "error_metric",
    "get_val_data_from_shuffled_train",
]


def visualize_results(
    metric_to_show: TypeMetrics,
    results_file_path: Path = RESULTS_PATH,
    filename: str = DEFAULT_RESULTS_FILENAME,
    or_filter_model_name: Optional[list[str]] = None,
) -> None:
    results_file_path = results_file_path / f"{filename}.csv"

    df_results = pd.read_csv(results_file_path, index_col=False)

    # Filter the model names
    df_results = apply_filter(df_results, or_filter_model_name)

    plt.figure(figsize=(10, 5))
    df_results_grouped = df_results.groupby(GROUPBY_COLUMNS)
    df_group_names = pd.DataFrame(
        list(df_results_grouped.indices.keys()), columns=GROUPBY_COLUMNS
    ).T
    df_group_names["No Change"] = df_group_names.apply(
        lambda x: x.nunique() == 1, axis=1
    )
    groupby_column_that_change = df_group_names[
        df_group_names["No Change"] == False
    ].index

    for group, df_group in df_results_grouped:
        x_axis_title = "Model name"
        for column_that_change in groupby_column_that_change:
            df_group["model_name"] += "_" + df_group[column_that_change].astype(str)
            x_axis_title += f" + {column_that_change}"

        ax = sns.boxplot(data=df_group, x="model_name", y=metric_to_show)

    ax.set_xlabel(x_axis_title)
    # Create a pretty string for the group info
    df_group_index_that_doesnt_change = df_group_names[df_group_names["No Change"]].loc[
        :, 0
    ]
    pretty_group = [
        f"{index_name}={df_group_index_that_doesnt_change[index_name]}"
        for index_name in df_group_index_that_doesnt_change.index
    ]
    # concat all elements of the list info a single string
    pretty_group = ", ".join(pretty_group)

    plt.title(f"{metric_to_show.upper()} for {pretty_group}")

    # Modify x-axis labels to be on multiple lines
    labels = [label.get_text().replace("_", "\n") for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    # the model on the x axis are a bit long, so we can add margin to the bottom
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def plot_loss_against_one_parameter(
    metric_to_show: TypeMetrics,
    parameter_to_show: str,
    or_filter_model_name: Optional[list[str]] = None,
    results_file_path: Path = RESULTS_PATH,
    filename: str = DEFAULT_RESULTS_FILENAME,
    include_scatter: bool = False,
    y_limits: Optional[list[int]] = None,
) -> None:
    """Scatter plot of the metric to show against one parameter. One line per model.

    Args:
        metric_to_show: The loss metric to show on the y-axis.
        parameter_to_show: The parameter to show on the x-axis.
        or_filter_model_name (optional): List of keywords to use to filter the model names. Defaults to None.
        results_file_path (optional): File path of the results file. Defaults to RESULTS_FILENAME.
        include_scatter (optional): Whether to include the scatter points. Defaults to False.
    """
    results_file_path = results_file_path / f"{filename}.csv"

    df_results = pd.read_csv(results_file_path, index_col=False)
    if parameter_to_show not in df_results.columns:
        raise ValueError(f"Parameter {parameter_to_show} is not in the results file.")

    # Filter the model names
    df_results = apply_filter(df_results, or_filter_model_name)

    GROUPBY_COLUMNS.remove(parameter_to_show)
    GROUPBY_COLUMNS.append("model_name")
    df_results_grouped = df_results.groupby(GROUPBY_COLUMNS)

    # Get a list of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_results_grouped)))  # type: ignore

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (group, df_group) in enumerate(df_results_grouped):
        averaged_df_group = (
            df_group[[metric_to_show, parameter_to_show]]
            .groupby(parameter_to_show)
            .mean()
        )
        ax.plot(
            averaged_df_group.index,
            averaged_df_group[metric_to_show],
            label=group[-1],
            color=colors[i],
        )
        if include_scatter:
            ax.scatter(
                df_group[parameter_to_show],
                df_group[metric_to_show],
                color=colors[i],
                alpha=0.5,
            )

    ax.legend()
    ax.set_xlabel(parameter_to_show)
    ax.set_ylabel(metric_to_show)
    if y_limits:
        if len(y_limits) != 2:
            raise ValueError(
                "The y_limits list should be of length 2. The first "
                "element is the maximum, the second is the minimum "
                "(e.g. [6000, 7000])"
            )
        ax.set_ylim(y_limits)
    ax.set_title(f"{metric_to_show.upper()} against {parameter_to_show}")
    plt.show()


def apply_filter(
    df_results: pd.DataFrame, or_filter_model_name: Optional[list[str]]
) -> pd.DataFrame:
    if or_filter_model_name:
        filtered_df = pd.DataFrame()
        for model_name_filter in or_filter_model_name:
            filtered_df = pd.concat(
                [
                    filtered_df,
                    df_results[
                        df_results["model_name"].str.contains(model_name_filter)
                    ],
                ]
            )
        return filtered_df
    return df_results


if __name__ == "__main__":
    # plot_loss_against_one_parameter(
    #     "rmse",
    #     "get_val_data_from_shuffled_train",
    #     ["FFN"],
    #     include_scatter=True,
    #     filename="results_ucsd_val_shuffle",
    #     # y_limits=[5400, 6000],
    # )
    visualize_results(
        "rmse", filename="results_ucsd_val_shuffle", or_filter_model_name=["FFNN"]
    )
