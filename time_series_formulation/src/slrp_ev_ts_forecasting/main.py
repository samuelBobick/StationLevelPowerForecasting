from slrp_ev_ts_forecasting.default_parameters import (
    DEFAULT_RESULTS_FILENAME,
    TypeModelChoice,
)
from slrp_ev_ts_forecasting.run_one_model import run_one_model

model_choice: TypeModelChoice = "KNN"

if __name__ == "__main__":
    for x_dim in [96 * 2]:
        for optimize_lags in ["short_opt", "long_opt"]:
            for i in range(2):
                run_one_model(
                    model_choice=model_choice,
                    model_parameters={
                        "x_dim": x_dim,
                        "optimize_lags": optimize_lags,
                    },  #
                    save_results_filename=DEFAULT_RESULTS_FILENAME,
                )
