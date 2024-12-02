from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest, validate_params
import pandas as pd
from darts import TimeSeries
from darts.models import NLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse, smape, r2_score, mape
import torch


class input_schema(BaseModel):
    input_chunk: int = Field(ge=1)
    output_chunk: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    n_epochs: int = Field(ge=1)


class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)

        # Save parameters to state
        self.input_chunk = params.input_chunk
        self.output_chunk = params.output_chunk
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.n_epochs = params.n_epochs

        self.model = None
        self.scaler_X = Scaler()
        self.scaler_y = Scaler()

    def __repr__(self):
        param_names = ['input_chunk', 'output_chunk', 'learning_rate', 'batch_size', 'n_epochs']
        param_string = ', '.join([f'{x}={self.__dict__[x]}' for x in param_names])
        return f'NLinearModel({param_string})'

    def fit(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."
        assert self.model is None, 'Model has already been trained.'

        # Prepare data
        input_series = TimeSeries.from_dataframe(features, fill_missing_dates=False, freq=None)
        target_series = TimeSeries.from_dataframe(labels.to_frame(), fill_missing_dates=False, freq=None)

        # Split data
        split_idx = int(len(input_series) * 0.8)
        input_train, input_val = input_series.split_before(split_idx)
        target_train, target_val = target_series.split_before(split_idx)

        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(input_train)
        y_train_scaled = self.scaler_y.fit_transform(target_train)
        X_val_scaled = self.scaler_X.transform(input_val)
        y_val_scaled = self.scaler_y.transform(target_val)

        # Define model
        self.model = NLinearModel(
            input_chunk_length=self.input_chunk,
            output_chunk_length=self.output_chunk,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={'lr': self.learning_rate},
            loss_fn=torch.nn.L1Loss(),
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
        )

        # Train model
        self.model.fit(
            series=y_train_scaled,
            past_covariates=X_train_scaled,
            val_series=y_val_scaled,
            val_past_covariates=X_val_scaled,
            verbose=True,
        )

    def predict(self, features):
        assert features is not None, "Features cannot be None."
        assert self.model is not None, "Model has not been trained yet."

        # Prepare input series
        input_series = TimeSeries.from_dataframe(features, fill_missing_dates=False, freq=None)

        # Scale features
        X_scaled = self.scaler_X.transform(input_series)

        # Generate predictions
        predictions_scaled = self.model.predict(n=len(X_scaled), past_covariates=X_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled)

    def score(self, features, labels):
        assert features is not None and labels is not None, "Features and labels cannot be None."
        predictions = self.predict(features)
        actual_series = TimeSeries.from_dataframe(labels.to_frame(), fill_missing_dates=False, freq=None)

        # Calculate evaluation metrics
        metrics = {
            "mae": mae(actual_series, predictions),
            "mse": mse(actual_series, predictions),
            "rmse": rmse(actual_series, predictions),
            "r2": r2_score(actual_series, predictions),
            "mape": mape(actual_series, predictions),
            "smape": smape(actual_series, predictions),
        }

        return metrics


class tests(base_unittest):
    def test_00_validate_input(self):
        """Test model instantiation with valid input parameters."""
        input_params = {
            "input_chunk": 48,
            "output_chunk": 1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "n_epochs": 100,
        }
        model = custom_model(input_params)
        self.assertEqual(model.input_chunk, 48, "Mismatch in input_chunk.")
        self.assertEqual(model.output_chunk, 1, "Mismatch in output_chunk.")
        self.assertEqual(model.learning_rate, 0.001, "Mismatch in learning_rate.")
        self.assertEqual(model.batch_size, 256, "Mismatch in batch_size.")
        self.assertEqual(model.n_epochs, 100, "Mismatch in n_epochs.")

    def test_01_runs_with_yaml_params(self):
        """Test model instantiation with dynamically loaded YAML parameters."""
        input_params = {
            "input_chunk": 36,
            "output_chunk": 2,
            "learning_rate": 0.005,
            "batch_size": 128,
            "n_epochs": 50,
        }
        model = custom_model(input_params)
        self.assertIsInstance(model, custom_model, "Failed to instantiate custom_model with YAML params.")

    def test_02_validate_model(self):
        """Test the fit, predict, and score functionality."""
        input_params = {
            "input_chunk": 48,
            "output_chunk": 1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "n_epochs": 10,
        }
        model = custom_model(input_params)

        # Load data
        df = pd.read_pickle("svr_feature_store_20000.pkl")
        target_column = 'label'
        features = df.drop(columns=[target_column])
        labels = df[target_column]

        # Train the model
        model.fit(features, labels)

        # Test prediction
        predictions = model.predict(features)
        self.assertIsNotNone(predictions, "Prediction output is None.")
        self.assertEqual(len(predictions), len(features), "Mismatch in prediction length.")

        # Test scoring
        metrics = model.score(features, labels)
        self.assertIn("mae", metrics, "MAE metric missing in score.")
        self.assertIn("mse", metrics, "MSE metric missing in score.")
        self.assertIn("rmse", metrics, "RMSE metric missing in score.")
        self.assertIn("r2", metrics, "R2 metric missing in score.")
        self.assertIn("mape", metrics, "MAPE metric missing in score.")
        self.assertIn("smape", metrics, "SMAPE metric missing in score.")

        # Print metrics for debugging
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
