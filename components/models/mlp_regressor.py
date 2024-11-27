import numpy as np
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
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
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
        assert self.model is None, 'Model has already been trained.'

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
    def test_00_runs_with_mock_params(self):
        """Test model instantiation with static mock parameters."""
        model = custom_model({
            'hidden_layer_sizes': (200, 100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 1000,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        })

        self.assertEqual(model.hidden_layer_sizes, (200, 100, 50), "Mismatch in hidden_layer_sizes")
        self.assertEqual(model.activation, "relu", "Mismatch in activation")
        self.assertEqual(model.solver, "adam", "Mismatch in solver")
        self.assertEqual(model.alpha, 0.001, "Mismatch in alpha")
        self.assertEqual(model.learning_rate, "adaptive", "Mismatch in learning_rate")
        self.assertEqual(model.max_iter, 1000, "Mismatch in max_iter")
        self.assertEqual(model.random_state, 42, "Mismatch in random_state")
        self.assertTrue(model.early_stopping, "Mismatch in early_stopping")
        self.assertEqual(model.validation_fraction, 0.1, "Mismatch in validation_fraction")

    def test_01_runs_with_yaml_params(self):
        """Test model instantiation with dynamically loaded YAML parameters."""
        custom_model(self.input_params)
