from common.testing import base_unittest
from pydantic import BaseModel, Field
from components.models.base_model import base_model
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class input_schema(BaseModel):
  iterations: int = Field(..., ge=1, description="The number of trees to be built.")
  learning_rate: float = Field(..., gt=0, description="The learning rate for the model.")
  depth: int = Field(..., ge=1, description="The depth of the trees.")
  l2_leaf_reg: float = Field(..., gt=0, description="L2 regularization coefficient.")
  loss_function: str = Field(..., description="The loss function to be used (e.g., 'RMSE', 'MAE').")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
  def __init__(self, input_params: dict):
      # Validate input parameters using the schema
      assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
      params = input_schema(**input_params)

      # Save parameters in the state
      self.iterations = params.iterations
      self.learning_rate = params.learning_rate
      self.depth = params.depth
      self.l2_leaf_reg = params.l2_leaf_reg
      self.loss_function = params.loss_function

      # Model starts as None
      self.model = None

  def __repr__(self):
      return f"""
      CatBoostRegressor(
          iterations={self.iterations},
          learning_rate={self.learning_rate},
          depth={self.depth},
          l2_leaf_reg={self.l2_leaf_reg},
          loss_function='{self.loss_function}'
      )
      """

  def fit(self, features, labels=None):
      assert self.model is None, 'A MODEL HAS ALREADY BEEN TRAINED'

      # Create and fit the CatBoost model
      catboost_model = CatBoostRegressor(
          iterations=self.iterations,
          learning_rate=self.learning_rate,
          depth=self.depth,
          l2_leaf_reg=self.l2_leaf_reg,
          loss_function=self.loss_function,
          verbose=0  # Suppress output during training
      )

      # Fit the model
      catboost_model.fit(features, labels)

      # Save the trained model
      self.model = catboost_model

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
          'r2': r2_score(labels, predictions)
      }

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
  def test_00_validate_input(self):
      """
      Test to validate the input parameters for the custom_model.
      """
      input_params = {
          "iterations": 100,
          "learning_rate": 0.1,
          "depth": 6,
          "l2_leaf_reg": 3,
          "loss_function": "RMSE"
      }
      model = custom_model(input_params)
      assert model.iterations == 100
      assert model.learning_rate == 0.1
      assert model.depth == 6
      assert model.l2_leaf_reg == 3
      assert model.loss_function == "RMSE"

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
          "iterations": 100,
          "learning_rate": 0.1,
          "depth": 6,
          "l2_leaf_reg": 3,
          "loss_function": "RMSE"
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
      assert 'r2' in scores