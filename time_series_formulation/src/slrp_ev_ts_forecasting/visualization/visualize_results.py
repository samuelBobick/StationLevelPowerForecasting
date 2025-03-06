from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from slrp_ev_ts_forecasting.default_parameters import (
    DEFAULT_RESULTS_FILENAME,
    RESULTS_PATH,
)

TypeMetrics = Literal[
    "rmse", "wrmse (alpha=2)", "mae", "wprmse (beta=3)", "r2", "elapsed_time"
]


def get_groupby_columns(df_results: pd.DataFrame) -> list[str]:
    groupby_columns = [
        col
        for col in df_results.columns
        if col
        not in [
            "date",
            "model_name",
            "rmse",
            "wrmse (alpha=2)",
            "wprmse (beta=3)",
            "mae",
            "r2",
            "error_std",
            "elapsed_time",
        ]
    ]
    return groupby_columns


def clean_str(string: str) -> str:
    return string.replace("_", "\n").capitalize()


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

    groupby_columns = get_groupby_columns(df_results)

    df_results_grouped = df_results.groupby(groupby_columns)
    df_group_names = pd.DataFrame(
        list(df_results_grouped.indices.keys()), columns=groupby_columns
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

        fig = px.box(
            df_group,
            x="model_name",
            y=metric_to_show,
            color="model_name",
            color_discrete_sequence=px.colors.qualitative.__dict__["Plotly"],
        )

    # Create a pretty string for the group info
    df_group_index_that_doesnt_change = df_group_names[df_group_names["No Change"]].loc[
        :, 0
    ]
    pretty_group = [
        f"{index_name}={df_group_index_that_doesnt_change[index_name]}"
        for index_name in df_group_index_that_doesnt_change.index
    ]
    # concat all elements of the list info a single string (and add some line breaks)
    item_counter = 0
    pretty_parameters_for_title = ""
    for item in pretty_group:
        if item_counter == 4:
            pretty_parameters_for_title += "<br>"
            item_counter = 0
        pretty_parameters_for_title += item + ", "
        item_counter += 1
    title = f"{metric_to_show.upper()} for {pretty_parameters_for_title}"

    # Modify x-axis labels to be on multiple lines
    tickvals = df_group["model_name"].unique()
    ticktext = [label.replace("_", "<br>") for label in tickvals]

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=metric_to_show,
        legend_title="Model Name",
        template="plotly_white",
        width=1300,  # Set the width of the plot
        height=600,  # Set the height of the plot
        showlegend=False,  # Show or hide the legend
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),  # Update x-axis labels
        margin=dict(l=0, r=0, t=150, b=0),  # Set the margins
    )

    fig.show()


def plot_loss_against_one_parameter(
    metric_to_show: TypeMetrics,
    parameter_to_show: str,
    second_metric_to_show: Optional[TypeMetrics] = None,
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

    groupby_columns = get_groupby_columns(df_results)

    groupby_columns.remove(parameter_to_show)
    groupby_columns.append("model_name")
    df_results_grouped = df_results.groupby(groupby_columns)

    # Get a list of colors
    # colors = plt.cm.viridis(np.linspace(0, 1, len(df_results_grouped)))  # type: ignore
    # colors = pc.qualitative.__dict__["Viridis"]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    for i, (group, df_group) in enumerate(df_results_grouped):
        averaged_df_group = (
            df_group[
                [metric_to_show, parameter_to_show]
                + ([second_metric_to_show] if second_metric_to_show else [])
            ]
            .groupby(parameter_to_show)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=averaged_df_group.index,
                y=averaged_df_group[metric_to_show],
                mode="lines+markers" if include_scatter else "lines",
                name=group[-1] + f" {metric_to_show}",
            ),
            secondary_y=False,
        )

        if second_metric_to_show:
            fig.add_trace(
                go.Scatter(
                    x=averaged_df_group.index,
                    y=averaged_df_group[second_metric_to_show],
                    mode="lines+markers" if include_scatter else "lines",
                    name=group[-1] + f" {second_metric_to_show}",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=f"{metric_to_show} vs {parameter_to_show}",
        xaxis_title=parameter_to_show,
        yaxis_title=clean_str(metric_to_show),
        yaxis2_title=clean_str(second_metric_to_show) if second_metric_to_show else "",
        legend_title="Model Name",
        template="plotly_white",
        width=700,
        height=400,
        showlegend=False,
    )
    fig.show()


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
    #     "x_dim",
    #     # second_metric_to_show="elapsed_time",
    #     or_filter_model_name=["XGBoost"],
    #     include_scatter=True,
    #     filename="experiment_lagOpt_202502",
    #     # y_limits=[5400, 6000],
    # )
    visualize_results("rmse", filename="experiment_basic_benchmark_202502")
