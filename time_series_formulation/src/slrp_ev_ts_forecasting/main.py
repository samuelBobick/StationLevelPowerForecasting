from slrp_ev_ts_forecasting.default_parameters import TypeModelChoice
from slrp_ev_ts_forecasting.run_one_model import run_one_model

model_choice: TypeModelChoice = "XGBoost"

if __name__ == "__main__":
    for x_dim in [96, 96 * 3, 96 * 5]:
        for optimize_lags in ["long_opt"]:
            for i in range(3):
                run_one_model(
                    model_choice=model_choice,
                    model_parameters={"x_dim": x_dim, "optimize_lags": optimize_lags},
                )
