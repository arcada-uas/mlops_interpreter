from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from components.models.base_model import base_model
from common.testing import base_unittest
import numpy as np


# Input schema for parameter validation
class input_schema(BaseModel):
    n_estimators: int = Field(100, ge=1, description="Number of trees in the XGBoost model.")
    max_depth: int = Field(6, ge=1, description="Maximum depth of the trees.")
    learning_rate: float = Field(0.3, gt=0, description="Learning rate for the model.")
    subsample: float = Field(1.0, gt=0, le=1, description="Subsample ratio of the training data.")
    colsample_bytree: float = Field(1.0, gt=0, le=1, description="Subsample ratio of columns when constructing each tree.")
    gamma: float = Field(0.0, ge=0, description="Minimum loss reduction required to make a further partition on a leaf node.")
    reg_alpha: float = Field(0.0, ge=0, description="L1 regularization term on weights.")
    reg_lambda: float = Field(1.0, ge=0, description="L2 regularization term on weights.")
    n_jobs: int = Field(-1, description="Number of parallel threads used to run XGBoost.")
    random_state: int = Field(42, description="Random seed for reproducibility.")


# XGBoost custom model implementation
class custom_model(base_model):
    def __init__(self, input_params: dict):
        assert isinstance(input_params, dict), f"Argument 'input_params' must be of type dict, got {type(input_params)}"
        params = input_schema(**input_params)

        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample
        self.colsample_bytree = params.colsample_bytree
        self.gamma = params.gamma
        self.reg_alpha = params.reg_alpha
        self.reg_lambda = params.reg_lambda
        self.n_jobs = params.n_jobs
        self.random_state = params.random_state

        self.model = None

    def __repr__(self):
        return f"""
        XGBRegressor(
            n_estimators={self.n_estimators},
            max_depth={self.max_depth},
            learning_rate={self.learning_rate},
            subsample={self.subsample},
            colsample_bytree={self.colsample_bytree},
            gamma={self.gamma},
            reg_alpha={self.reg_alpha},
            reg_lambda={self.reg_lambda},
            n_jobs={self.n_jobs},
            random_state={self.random_state}
        )
        """

    def fit(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."
        assert self.model is None, "Model has already been trained."

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model.fit(features, labels)

    def predict(self, features):
        assert self.model is not None, "Model has not been trained yet."
        assert features is not None, "Features cannot be None."
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


# Unit tests for custom_model
class tests(base_unittest):
    def test_00_validate_input(self):
        """Test model instantiation with valid input parameters."""
        input_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'random_state': 42
        }
        model = custom_model(input_params)
        self.assertEqual(model.n_estimators, 100, "Mismatch in n_estimators.")
        self.assertEqual(model.max_depth, 6, "Mismatch in max_depth.")
        self.assertEqual(model.learning_rate, 0.3, "Mismatch in learning_rate.")
        self.assertEqual(model.subsample, 0.8, "Mismatch in subsample.")
        self.assertEqual(model.colsample_bytree, 0.8, "Mismatch in colsample_bytree.")
        self.assertEqual(model.gamma, 0.0, "Mismatch in gamma.")
        self.assertEqual(model.reg_alpha, 0.0, "Mismatch in reg_alpha.")
        self.assertEqual(model.reg_lambda, 1.0, "Mismatch in reg_lambda.")
        self.assertEqual(model.n_jobs, -1, "Mismatch in n_jobs.")
        self.assertEqual(model.random_state, 42, "Mismatch in random_state.")

    def test_01_runs_with_yaml_params(self):
        """Test model instantiation with dynamically loaded YAML parameters."""
        custom_model(self.input_params)
        
    def test_012_train_and_evaluate(self):
        """Test to train and evaluate the custom_model."""
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'random_state': 42
        }

        model = custom_model(input_params)
        model.fit(X_train, y_train)
        scores = model.score(X_test, y_test)

        self.assertIn("mae", scores, "MAE missing from results.")
        self.assertIn("mse", scores, "MSE missing from results.")
        self.assertIn("rmse", scores, "RMSE missing from results.")
        self.assertIn("r2", scores, "R2 missing from results.")
        self.assertIn("mape", scores, "MAPE missing from results.")
        self.assertIn("smape", scores, "SMAPE missing from results.")
        print("Evaluation Metrics:", scores)
