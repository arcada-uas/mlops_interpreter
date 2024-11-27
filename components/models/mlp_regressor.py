import numpy as np
import yaml

from pydantic import BaseModel, Field
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from components.models.base_model import base_model
from common.testing import base_unittest


class InputSchema(BaseModel):
    hidden_layer_sizes: tuple = Field((200, 100, 50), description="Number of neurons in each hidden layer.")
    activation: str = Field("relu", description="Activation function for the hidden layers.")
    solver: str = Field("adam", description="Optimization solver.")
    alpha: float = Field(0.001, gt=0, description="Regularization term (L2 penalty).")
    learning_rate: str = Field("adaptive", description="Learning rate schedule for optimization.")
    max_iter: int = Field(1000, ge=1, description="Maximum number of iterations.")
    random_state: int = Field(42, description="Random state for reproducibility.")
    early_stopping: bool = Field(True, description="Enable early stopping based on validation.")
    validation_fraction: float = Field(0.1, gt=0, le=1, description="Proportion of validation set during training.")


class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = InputSchema(**input_params)

        self.hidden_layer_sizes = params.hidden_layer_sizes
        self.activation = params.activation
        self.solver = params.solver
        self.alpha = params.alpha
        self.learning_rate = params.learning_rate
        self.max_iter = params.max_iter
        self.random_state = params.random_state
        self.early_stopping = params.early_stopping
        self.validation_fraction = params.validation_fraction

        self.model = None

    def __repr__(self):
        return (
            f"MLPRegressor("
            f"hidden_layer_sizes={self.hidden_layer_sizes}, "
            f"activation='{self.activation}', "
            f"solver='{self.solver}', "
            f"alpha={self.alpha}, "
            f"learning_rate='{self.learning_rate}', "
            f"max_iter={self.max_iter}, "
            f"random_state={self.random_state}, "
            f"early_stopping={self.early_stopping}, "
            f"validation_fraction={self.validation_fraction})"
        )

    def fit(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."

        if self.model is not None:
            raise RuntimeError("Model has already been trained.")
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction
        )
        self.model.fit(features, labels)

    def predict(self, features):
        assert features is not None, "Features cannot be None."
        assert self.model is not None, "Model has not been trained yet."
        return self.model.predict(features)

    def score(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert self.model is not None, "Model has not been trained yet."

        predictions = self.model.predict(features)

        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)
        mape = np.mean(np.abs((labels - predictions) / labels)) * 100
        smape = 100 * np.mean(2 * np.abs(labels - predictions) / (np.abs(labels) + np.abs(predictions)))

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "smape": smape
        }


class tests(base_unittest):
    def test_00_validate_input(self):
        with open("model_config.yaml", "r") as file:
            input_params = yaml.safe_load(file)["params"]

        model = custom_model(input_params)
        assert model.hidden_layer_sizes == (200, 100, 50)
        assert model.activation == "relu"
        assert model.solver == "adam"
        assert model.alpha == 0.001
        assert model.learning_rate == "adaptive"
        assert model.max_iter == 1000
        assert model.random_state == 42
        assert model.early_stopping is True
        assert model.validation_fraction == 0.1

    def test_01_train_and_evaluate(self):
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open("model_config.yaml", "r") as file:
            input_params = yaml.safe_load(file)["params"]

        model = custom_model(input_params)
        model.fit(X_train, y_train)
        scores = model.score(X_test, y_test)
        assert "mae" in scores
        assert "mse" in scores
        assert "rmse" in scores
        assert "r2" in scores

    def test_02_corner_cases(self):
        with open("model_config.yaml", "r") as file:
            input_params = yaml.safe_load(file)["params"]

        # Zero samples case
        X, y = np.empty((0, 10)), np.empty((0,))
        model = custom_model(input_params)
        try:
            model.fit(X, y)
        except ValueError:
            pass  # Expected

        # Invalid input types
        X, y = "invalid_input", [1, 2, 3]
        try:
            model.fit(X, y)
        except ValueError:
            pass  # Expected

    def test_03_consistency_with_random_state(self):
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open("model_config.yaml", "r") as file:
            input_params = yaml.safe_load(file)["params"]

        model1 = custom_model(input_params)
        model1.fit(X_train, y_train)
        scores1 = model1.score(X_test, y_test)

        model2 = custom_model(input_params)
        model2.fit(X_train, y_train)
        scores2 = model2.score(X_test, y_test)

        assert scores1 == scores2, "Scores should be consistent with the same random_state."
