import numpy as np
from pydantic import BaseModel, Field
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from components.models.base_model import base_model
from common.testing import base_unittest


class input_schema(BaseModel):
    C: float = Field(1.0, gt=0, description="Regularization parameter for SVR.")
    epsilon: float = Field(0.1, ge=0, description="Epsilon-tube within which no penalty is associated in the training loss function.")
    kernel: str = Field("rbf", description="Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid').")
    degree: int = Field(3, ge=1, description="Degree of the polynomial kernel function ('poly'). Ignored by other kernels.")
    gamma: str = Field("scale", description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.")
    coef0: float = Field(0.0, description="Independent term in kernel function ('poly' and 'sigmoid').")
    shrinking: bool = Field(True, description="Whether to use the shrinking heuristic.")
    tol: float = Field(1e-3, gt=0, description="Tolerance for stopping criteria.")
    max_iter: int = Field(-1, description="Maximum number of iterations (-1 for no limit).")


class custom_model(base_model):
    def __init__(self, input_params: dict):
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
        params = input_schema(**input_params)

        self.C = params.C
        self.epsilon = params.epsilon
        self.kernel = params.kernel
        self.degree = params.degree
        self.gamma = params.gamma
        self.coef0 = params.coef0
        self.shrinking = params.shrinking
        self.tol = params.tol
        self.max_iter = params.max_iter

        self.model = None

    def __repr__(self):
        return f"""
        SVR(
            C={self.C},
            epsilon={self.epsilon},
            kernel='{self.kernel}',
            degree={self.degree},
            gamma='{self.gamma}',
            coef0={self.coef0},
            shrinking={self.shrinking},
            tol={self.tol},
            max_iter={self.max_iter}
        )
        """

    def fit(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."
        assert self.model is None, "A MODEL HAS ALREADY BEEN TRAINED."

        self.model = SVR(
            C=self.C,
            epsilon=self.epsilon,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            max_iter=self.max_iter
        )
        self.model.fit(features, labels)

    def predict(self, features):
        assert features is not None, "Features cannot be None."
        assert self.model is not None, "A MODEL HAS NOT BEEN TRAINED YET."
        return self.model.predict(features)

    def score(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert self.model is not None, "A MODEL HAS NOT BEEN TRAINED YET."

        predictions = self.model.predict(features)

        return {
            'mae': mean_absolute_error(labels, predictions),
            'mse': mean_squared_error(labels, predictions),
            'rmse': np.sqrt(mean_squared_error(labels, predictions)),
            'r2': r2_score(labels, predictions),
            'mape': np.mean(np.abs((labels - predictions) / labels)) * 100,
            'smape': 100 * np.mean(2 * np.abs(labels - predictions) / (np.abs(labels) + np.abs(predictions)))
        }


class tests(base_unittest):
    def test_00_runs_with_mock_params(self):
        """Test model instantiation with static mock parameters."""
        model = custom_model({
            'C': 1.0,
            'epsilon': 0.1,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'tol': 1e-3,
            'max_iter': -1
        })

        self.assertEqual(model.C, 1.0, "Mismatch in C.")
        self.assertEqual(model.epsilon, 0.1, "Mismatch in epsilon.")
        self.assertEqual(model.kernel, 'rbf', "Mismatch in kernel.")
        self.assertEqual(model.degree, 3, "Mismatch in degree.")
        self.assertEqual(model.gamma, 'scale', "Mismatch in gamma.")
        self.assertEqual(model.coef0, 0.0, "Mismatch in coef0.")
        self.assertEqual(model.shrinking, True, "Mismatch in shrinking.")
        self.assertEqual(model.tol, 1e-3, "Mismatch in tol.")
        self.assertEqual(model.max_iter, -1, "Mismatch in max_iter.")

    def test_01_runs_with_yaml_params(self):
        """Test model instantiation with dynamically loaded YAML parameters."""
        custom_model(self.input_params)

    def test_02_train_and_evaluate(self):
        """Test to train and evaluate the custom_model with mock data."""
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = custom_model({
            'C': 1.0,
            'epsilon': 0.1,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'tol': 1e-3,
            'max_iter': -1
        })

        model.fit(X_train, y_train)
        scores = model.score(X_test, y_test)

        self.assertIn('mae', scores, "MAE not found in scores.")
        self.assertIn('mse', scores, "MSE not found in scores.")
        self.assertIn('rmse', scores, "RMSE not found in scores.")
        self.assertIn('r2', scores, "R2 not found in scores.")
        self.assertIn('mape', scores, "MAPE not found in scores.")
        self.assertIn('smape', scores, "SMAPE not found in scores.")

        print("Evaluation Metrics:", scores)
