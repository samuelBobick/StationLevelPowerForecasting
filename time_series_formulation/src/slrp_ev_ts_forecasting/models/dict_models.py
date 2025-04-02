from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from slrp_ev_ts_forecasting.default_parameters import TypeModelChoice
from slrp_ev_ts_forecasting.models.ffnn import FFNN
from slrp_ev_ts_forecasting.models.knn import KNN
from slrp_ev_ts_forecasting.models.last_week import LastWeek
from slrp_ev_ts_forecasting.models.linear_model import LinearModel
from slrp_ev_ts_forecasting.models.lstm import LSTM
from slrp_ev_ts_forecasting.models.peak_persistence import PeakPersistence
from slrp_ev_ts_forecasting.models.session_naive import SessionNaive
from slrp_ev_ts_forecasting.models.similar_day import SimilarDay
from slrp_ev_ts_forecasting.models.sktime_base import SktimeBaseModel
from slrp_ev_ts_forecasting.models.tcn import TCN
from slrp_ev_ts_forecasting.models.xgboost_model import XGBoost

DICT_MODEL: dict[TypeModelChoice, dict] = {
    "LinearRegression": {
        "model": LinearModel,
        "model_params": {},
        "fit_params": {},
    },
    "KNN": {
        "model": KNN,
        "model_params": {},
        "fit_params": {},
    },
    "XGBoost": {
        "model": XGBoost,
        "model_params": {},
        "fit_params": {},
    },
    "Last_Week": {"model": LastWeek, "model_params": {}, "fit_params": {}},
    "Similar_Day": {"model": SimilarDay, "model_params": {}, "fit_params": {}},
    "Basic_NN": {
        "model": FFNN,
        "model_params": {},
        "fit_params": {},
    },
    "LSTM": {
        "model": LSTM,
        "model_params": {},
        "fit_params": {},
    },
    "TCN": {
        "model": TCN,
        "model_params": {},
        "fit_params": {},
    },
    "AutoETS": {
        "model": SktimeBaseModel,
        "model_params": {
            "forecaster": AutoETS(auto=True, sp=24, n_jobs=-1, maxiter=20),
            "include_exogenous": True,
            "downsample_hours": 1,
            "refit_model_before_predictions": False,
        },  # default maxiter = 1000
        "fit_params": {},
    },
    "AutoARIMA": {
        # takes a very long time to run...
        "model": SktimeBaseModel,
        "model_params": {
            "forecaster": AutoARIMA(
                sp=int(96 / (4 * 4)),
                out_of_sample_size=int(96 / (4 * 4) * 7),
                maxiter=15,
                n_jobs=-1,
                start_params=[1, 1],
                max_order=7,
                seasonal=True,
                stationary=True,
            ),
            "include_exogenous": True,
            "downsample_hours": 4,
            "refit_model_before_predictions": False,
            "start_data_date": "2023-08",
        },  # default maxiter = 1000
        "fit_params": {},
    },
    "ARIMA": {
        "model": SktimeBaseModel,
        "model_params": {
            "forecaster": ARIMA(
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 96 * 7 / (4 * 2)),
            ),
            "include_exogenous": False,
            "downsample_hours": 2,
            "refit_model_before_predictions": False,
            "start_data_date": "2023-01",
        },  # default maxiter = 1000
        "fit_params": {},
    },
    "Prophet": {
        "model": SktimeBaseModel,
        "model_params": {
            "forecaster": Prophet(
                seasonality_mode="additive",
                n_changepoints=40,
                # add_country_holidays={"country_name": "US"},
                yearly_seasonality=False,  # type: ignore
                weekly_seasonality=True,  # type: ignore
                daily_seasonality=True,  # type: ignore
                # the three growth arguments go together.
                # They make the model slightly more precise (by a few percents)
                # but also much slower to make predictions
                # Don't forget that the data is normalized (btw 0 and 1)
                # growth_floor=0,
                # growth_cap=1,
                # growth="logistic",
            ),
            "include_exogenous": False,
            "downsample_hours": 2,
            "refit_model_before_predictions": False,
            # "start_data_date": "2023",
        },
        "fit_params": {},
    },
    "PeakPersistence": {
        "model": PeakPersistence,
        "model_params": {},
        "fit_params": {},
    },
    "SessionNaive": {
        "model": SessionNaive,
        "model_params": {},
        "fit_params": {},
    },
}
