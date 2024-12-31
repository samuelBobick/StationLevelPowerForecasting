import json
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import slrp_ev_ts_forecasting.default_parameters as default_parameters
from sklearn import linear_model
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel


class LinearModel(RegressionBaseModel):

    def __init__(
        self,
        x_dim=default_parameters.X_DIM,
        lookahead=default_parameters.LOOKAHEAD,
        alpha=default_parameters.ALPHA,
        time_mode: Literal["window", "cyclical"] = default_parameters.TIME_MODE,
        optimize_lags: default_parameters.TypeOptimizeLags = default_parameters.OPTIMIZE_LAGS,
        get_val_data_from_shuffled_train: bool = default_parameters.GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
    ):
        """_summary_

        Args:
            x_dim (int, optional): How many past timesteps ahead we want to use as inputs. Defaults to 16.
            lookahead (int, optional): How many timesteps ahead we want to predict. Defaults to 16.
            alpha (int, optional): Underpredictions are penalized alpha times more than overpredictions for weighted error metric. Defaults to 2.
        """
        super().__init__(
            x_dim=x_dim,
            lookahead=lookahead,
            alpha=alpha,
            time_mode=time_mode,
            optimize_lags=optimize_lags,
            get_val_data_from_shuffled_train=get_val_data_from_shuffled_train,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
        )
        self.alpha = alpha
        self.time_mode = time_mode
        # TODO: Do no forget to normalize the "next user power profile", dividing it by the max power (6.6)

    @property
    def model_str_name(self):
        return "LinearModel" + self.model_str_name_suffix

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_mask: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        val_mask: pd.Series,
    ):
        X_input = pd.concat(
            [
                X_train[train_mask].drop(
                    self.cols_to_drop_for_model,
                    axis=1,
                ),
                X_val[val_mask].drop(
                    self.cols_to_drop_for_model,
                    axis=1,
                ),
            ],
            axis=0,
        )
        self.feature_names = list(X_input.columns)

        y_input = pd.concat([y_train[train_mask], y_val[val_mask]], axis=0)
        self.label_names = list(y_input.columns)

        lm = linear_model.LinearRegression()
        lm.fit(X_input, y_input)
        return lm

    def predict_model(self, model, X_test: pd.DataFrame):
        return model.predict(X_test)

    def save_model(self, model, model_name: str, plot_feature_importance: bool = False):
        saved_model_filename = (
            default_parameters.SAVED_MODELS_PATH / f"{model_name}.json"
        )
        if plot_feature_importance:
            self.plot_feature_importance(model)

        # Extract parameters
        params = {
            "intercept": (
                model.intercept_.tolist()
                if hasattr(model.intercept_, "tolist")
                else model.intercept_
            ),
            "coefficients": model.coef_.tolist(),
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }

        # Save to JSON file
        with open(saved_model_filename, "w") as json_file:
            json.dump(params, json_file, indent=4)

    def plot_feature_importance(self, model):
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": abs(model.coef_[0])}
        )

        # Sort and plot
        importance_df.sort_values(by="Importance", ascending=False, inplace=True)

        fig = go.Figure(
            data=[go.Bar(x=importance_df["Feature"], y=importance_df["Importance"])]
        )
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Coefficient Absolute Value",
            title=f"Feature Importance in Linear Regression for label {self.label_names[0]}",
            xaxis=dict(tickangle=-90),
        )

        fig.show()
