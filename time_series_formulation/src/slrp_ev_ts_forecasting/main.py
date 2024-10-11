from slrp_ev_ts_forecasting.default_parameters import TypeModelChoice
from slrp_ev_ts_forecasting.run_one_model import run_one_model

model_choice: TypeModelChoice = "Similar_Day"

if __name__ == "__main__":
    run_one_model(model_choice=model_choice)
