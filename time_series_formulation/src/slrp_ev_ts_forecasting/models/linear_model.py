import json
from typing import Literal

import numpy as np
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
        scaling_mode: default_parameters.TypeScalingMode = default_parameters.SCALING_MODE,
        scaling_parameters: tuple | pd.DataFrame | None = None,
        session_based_mode: bool = default_parameters.SESSION_BASED_MODE,
        peak_prediction: bool = default_parameters.PEAK_PREDICTION,
        add_number_of_sessions: bool = default_parameters.ADD_NUMBER_OF_SESSIONS,
        add_fraction_of_regular_sessions: bool = default_parameters.ADD_FRACTION_OF_REGULAR_SESSIONS,
        use_all_active_sessions: bool = default_parameters.USE_ALL_ACTIVE_SESSIONS,
        number_of_artificial_datasets: int = default_parameters.NUMBER_OF_ARTIFICIAL_DATASETS,
        random_start_time: bool = default_parameters.RANDOM_START_TIME,
        shuffle_power_profiles: bool = default_parameters.SHUFFLE_POWER_PROFILES,
        random_power_profile_shapes: bool = default_parameters.RANDOM_POWER_PROFILE_SHAPES,
        random_user_needs: bool = default_parameters.RANDOM_USER_NEEDS,
        random_choices: bool = default_parameters.RANDOM_CHOICES,
        add_number_of_evses_available: bool = default_parameters.ADD_NUMBER_OF_EVSES_AVAILABLE,
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
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
            session_based_mode=session_based_mode,
            peak_prediction=peak_prediction,
            add_number_of_sessions=add_number_of_sessions,
            add_fraction_of_regular_sessions=add_fraction_of_regular_sessions,
            use_all_active_sessions=use_all_active_sessions,
            number_of_artificial_datasets=number_of_artificial_datasets,
            random_start_time=random_start_time,
            shuffle_power_profiles=shuffle_power_profiles,
            random_power_profile_shapes=random_power_profile_shapes,
            random_user_needs=random_user_needs,
            random_choices=random_choices,
            add_number_of_evses_available=add_number_of_evses_available,
        )
        self.alpha = alpha
        self.time_mode = time_mode
        # TODO: Do no forget to normalize the "next user power profile", dividing it by the max power (6.6)
        if default_parameters.RANDOM_SEED is not None:
            self.rs = np.random.RandomState(self.rng.bit_generator._seed_seq.entropy)  # type: ignore
        else:
            self.rs = None

        # get all the parameters of the class
        # and put them in a dict for when we save the model
        parameters_to_not_save = [
            "verbose",
            "rng",
            "pacf_top_values",
            "cols_to_drop_for_model",
            "rs",
        ]
        self.parameters_dict = {
            k: v
            for k, v in vars(self).items()
            if (
                not k.startswith("_")
                and k not in parameters_to_not_save
                and type(v) in [int, float, str, bool, list, dict]
            )
        }

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

        if self.peak_prediction:
            # for peak prediction, the model is faster to train
            # and we predict a single value
            # ElasticNet, which combines L1 and L2 regularization, would give the best results
            # however, since we use this model in the optimizer,
            # we do not want too many values to be equal to zero
            # therefore, we go with Ridge regression, which reduces the
            # coefficients, but does not set them to zero
            lm = linear_model.RidgeCV(
                alphas=np.array([0.1, 0.5, 1.0]),
                # We could go higher to 5.0, 10.0 but we avoid, for more stability in the optimizer
            )
            # for peak_prediction, we only predict a single value, so to avoid
            # a warning, we convert the y_input to a 1D array
            y_input = y_input["peak_power"]
        else:
            # for multi-output modes, we cannot have as many alphas
            # in the CV search
            lm = linear_model.MultiTaskElasticNetCV(random_state=self.rs, n_alphas=10)
            # lm = linear_model.LinearRegression()
        lm.fit(X_input, y_input)

        # print alpha and l1 parameters chosen for elastic net
        if self.peak_prediction:
            print(f"alpha: {lm.alpha_}")

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
            "model_parameters": self.parameters_dict,
            "intercept": (
                model.intercept_.tolist()
                if hasattr(model.intercept_, "tolist")
                else model.intercept_
            ),
            "coefficients": model.coef_.tolist(),
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }

        # make sure that the intercept is a list (even if it is a single value, as for peak prediction)
        if isinstance(params["intercept"], float):
            params["intercept"] = [params["intercept"]]

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
