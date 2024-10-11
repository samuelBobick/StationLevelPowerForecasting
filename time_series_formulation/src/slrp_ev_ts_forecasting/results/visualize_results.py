from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from slrp_ev_ts_forecasting.default_parameters import RESULTS_FILENAME


def visualize_results(
    metric_to_show: Literal["rmse", "wrmse (alpha=2)", "mae", "wprmse (beta=3)", "r2"],
    results_file_path: Path = RESULTS_FILENAME,
) -> None:
    df_results = pd.read_csv(results_file_path, index_col=False)
    group_by_columns = [
        "batch_size",
        "x_dim",
        "lookahead",
        "time_mode",
        "dataset",
        "error_metric",
    ]

    df_results_grouped = df_results.groupby(group_by_columns)
    for group, df_group in df_results_grouped:
        plt.figure(figsize=(10, 5))
        ax = sns.boxplot(data=df_group, x="model_name", y=metric_to_show)

        # Create a pretty string for the group info
        pretty_group = [
            f"{group_name}={group_value}"
            for group_name, group_value in zip(group_by_columns, group)
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


if __name__ == "__main__":
    visualize_results("r2")
