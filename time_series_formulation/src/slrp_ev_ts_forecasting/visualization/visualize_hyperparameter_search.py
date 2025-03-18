from pathlib import Path
from typing import Optional

import pandas as pd
from plotly.subplots import make_subplots
from slrp_ev_ts_forecasting.compute_losses import TypeMetrics
from slrp_ev_ts_forecasting.default_parameters import (
    DEFAULT_RESULTS_FILENAME,
    RESULTS_PATH,
)
from slrp_ev_ts_forecasting.utils.utils_visualization import (
    apply_filter,
    get_groupby_columns,
    parse_model_names,
)
from slrp_ev_ts_forecasting.visualization.visualize_results import (
    plot_loss_against_one_parameter,
)


def visualize_hyperparameter_search_with_subplots(
    metric_to_show: TypeMetrics,
    results_file_path: Path = RESULTS_PATH,
    filename: str = DEFAULT_RESULTS_FILENAME,
    or_filter_model_name: Optional[list[str]] = None,
) -> None:
    results_file_path = results_file_path / f"{filename}.csv"

    df_results = pd.read_csv(results_file_path, index_col=False)
    df_results = parse_model_names(df_results)
    # Filter the model names
    df_results = apply_filter(df_results, or_filter_model_name)

    model_names = df_results["model_name"].unique()
    if len(model_names) > 1:
        raise ValueError(
            "This function is intended to be used when all models are the same "
            "only the hyperparameters should change. If you have different models, "
            "please use visualize_results instead."
        )
    model_name = model_names[0]

    groupby_columns = get_groupby_columns(df_results)

    # Remove columns where there is only one unique value
    df_group_names = pd.DataFrame(
        list(df_results.groupby(groupby_columns).indices.keys()),
        columns=groupby_columns,
    ).T
    df_group_names["No Change"] = df_group_names.apply(
        lambda x: x.nunique() == 1, axis=1
    )
    parameters_to_show = df_group_names[
        df_group_names["No Change"] == False
    ].index.tolist()

    if parameters_to_show == []:
        print(
            "There are no parameters to show. This means that all the models in this file"
            "have the same hyperparameters."
        )
        return

    # Calculate the number of rows needed for 3 plots per row
    num_rows = (len(parameters_to_show) + 2) // 3

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=3,
        subplot_titles=parameters_to_show,
        specs=[[{"secondary_y": True}] * 3] * num_rows,
        shared_yaxes="all",
    )

    for i, parameter in enumerate(parameters_to_show):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig = plot_loss_against_one_parameter(
            metric_to_show=metric_to_show,
            parameter_to_show=parameter,
            results_file_path=results_file_path.parent,
            second_metric_to_show="elapsed_time",
            filename=filename,
            separate_models=False,
            include_scatter=False,
            fig=fig,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Hyperparameter Search {metric_to_show} Results for {model_name}",
        height=400 * num_rows,  # Adjust height based on number of subplots
        showlegend=False,
    )

    fig.show()


if __name__ == "__main__":
    visualize_hyperparameter_search_with_subplots(
        metric_to_show="rmse",
        filename="hyperparameter_search_XGBoost_initial_models_v2",
    )
