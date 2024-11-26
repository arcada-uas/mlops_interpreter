from common.testing import base_unittest
from pydantic import BaseModel, Field
from components.models.base_model import base_model
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class input_schema(BaseModel):
    C: float = Field(..., gt=0, description="Regularization parameter for SVR.")
    epsilon: float = Field(..., ge=0, description="Epsilon-tube within which no penalty is associated in the training loss function.")
    kernel: str = Field(..., description="Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid').")
    n_splits: int = Field(3, ge=2, description="Number of splits for time series cross-validation.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):
        # Validate input parameters using the schema
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
        params = input_schema(**input_params)

        # Save parameters in the state
        self.C = params.C
        self.epsilon = params.epsilon
        self.kernel = params.kernel
        self.n_splits = params.n_splits

        # Model starts as None
        self.model = None

    def __repr__(self):
        return f"""
        SVR(
            C={self.C},
            epsilon={self.epsilon},
            kernel='{self.kernel}'
        )
        """

    def fit(self, features, labels=None):
        assert self.model is None, 'A MODEL HAS ALREADY BEEN TRAINED'

        # Create a pipeline with a scaler and SVR model
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel))
        ])

        # Fit the pipeline
        svr_pipeline.fit(features, labels)

        # Save the trained model
        self.model = svr_pipeline

    def predict(self, features):
        assert self.model is not None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)

    def score(self, features, labels):
        assert self.model is not None, 'A MODEL HAS NOT BEEN TRAINED YET'

        # Predict on features
        predictions = self.model.predict(features)

        # Compute metrics directly
        return {
            'mae': mean_absolute_error(labels, predictions),
            'mse': mean_squared_error(labels, predictions),
            'rmse': np.sqrt(mean_squared_error(labels, predictions)),
            'r2': r2_score(labels, predictions),
            'mape': np.mean(np.abs((labels - predictions) / labels)) * 100,
            'mase': float(mean_absolute_error(labels, predictions)) / float(np.mean(np.abs(np.array(labels[1:]) - np.array(labels[:-1])))),
            'smape': 100 * np.mean(2 * np.abs(labels - predictions) / (np.abs(labels) + np.abs(predictions)))
        }

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        """
        Test to validate the input parameters for the custom_model.
        """
        input_params = {
            "C": 1.0,
            "epsilon": 0.1,
            "kernel": "rbf",
            "n_splits": 3
        }
        model = custom_model(input_params)
        assert model.C == 1.0
        assert model.epsilon == 0.1
        assert model.kernel == "rbf"
        assert model.n_splits == 3

    def test_01_train_and_evaluate(self):
        """
        Test to train and evaluate the custom_model.
        """
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        # Generate synthetic data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define input parameters
        input_params = {
            "C": 1.0,
            "epsilon": 0.1,
            "kernel": "rbf",
            "n_splits": 3
        }

        # Initialize and train the model
        model = custom_model(input_params)
        model.fit(X_train, y_train)

        # Evaluate the model
        scores = model.score(X_test, y_test)
        print(f"Evaluation Metrics: {scores}")
        assert 'mae' in scores
        assert 'mse' in scores
        assert 'rmse' in scores
        assert 'r2_score' in scores
        assert 'mape' in scores
        assert 'mase' in scores
        assert 'smape' in scores
