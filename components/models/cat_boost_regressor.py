import numpy as np
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from components.models.base_model import base_model
from common.testing import base_unittest


class InputSchema(BaseModel):
    iterations: int = Field(100, ge=1, description="The number of trees to be built.")
    learning_rate: float = Field(0.1, gt=0, description="The learning rate for the model.")
    depth: int = Field(6, ge=1, description="The depth of the trees.")
    l2_leaf_reg: float = Field(3.0, gt=0, description="L2 regularization coefficient.")
    loss_function: str = Field("RMSE", description="The loss function to be used.")
    random_seed: int = Field(42, description="Random state for reproducibility.")
    verbose: int = Field(0, description="Verbose level for CatBoost.")


class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = InputSchema(**input_params)

        self.iterations = params.iterations
        self.learning_rate = params.learning_rate
        self.depth = params.depth
        self.l2_leaf_reg = params.l2_leaf_reg
        self.loss_function = params.loss_function
        self.random_seed = params.random_seed
        self.verbose = params.verbose

        self.model = None

    def __repr__(self):
        return (
            f"CatBoostRegressor("
            f"iterations={self.iterations}, "
            f"learning_rate={self.learning_rate}, "
            f"depth={self.depth}, "
            f"l2_leaf_reg={self.l2_leaf_reg}, "
            f"loss_function='{self.loss_function}', "
            f"random_seed={self.random_seed}, "
            f"verbose={self.verbose})"
        )

    def fit(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."
        assert self.model is None, 'Model has already been trained.'
        
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose
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
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0
        })

        self.assertEqual(model.iterations, 100, "Mismatch in iterations")
        self.assertEqual(model.learning_rate, 0.1, "Mismatch in learning_rate")
        self.assertEqual(model.depth, 6, "Mismatch in depth")
        self.assertEqual(model.l2_leaf_reg, 3.0, "Mismatch in l2_leaf_reg")
        self.assertEqual(model.loss_function, "RMSE", "Mismatch in loss_function")
        self.assertEqual(model.random_seed, 42, "Mismatch in random_seed")
        self.assertEqual(model.verbose, 0, "Mismatch in verbose")

    def test_01_runs_with_yaml_params(self):
            """Test model instantiation with dynamically loaded YAML parameters."""
            custom_model(self.input_params)
        
    def test_02_train_and_evaluate(self):
        """Test to train and evaluate the custom_model with mock data."""
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = custom_model({
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0
        })

        model.fit(X_train, y_train)
        scores = model.score(X_test, y_test)

        self.assertTrue("mae" in scores, "MAE not found in scores")
        self.assertTrue("mse" in scores, "MSE not found in scores")
        self.assertTrue("rmse" in scores, "RMSE not found in scores")
        self.assertTrue("r2" in scores, "R2 not found in scores")
        self.assertTrue("mape" in scores, "MAPE not found in scores")
        self.assertTrue("smape" in scores, "SMAPE not found in scores")

        print("Evaluation Metrics:", scores)
