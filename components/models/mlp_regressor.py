from common.testing import base_unittest
from pydantic import BaseModel, Field
from components.models.base_model import base_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class input_schema(BaseModel):
  hidden_layer_sizes: tuple = Field((100,), description="The ith element represents the number of neurons in the ith hidden layer.")
  activation: str = Field('relu', description="Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu').")
  solver: str = Field('adam', description="The solver for weight optimization ('lbfgs', 'sgd', 'adam').")
  alpha: float = Field(0.0001, gt=0, description="L2 penalty (regularization term) parameter.")
  learning_rate: str = Field('constant', description="Learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive').")
  max_iter: int = Field(200, ge=1, description="Maximum number of iterations.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
  def __init__(self, input_params: dict):
      # Validate input parameters using the schema
      assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
      params = input_schema(**input_params)

      # Save parameters in the state
      self.hidden_layer_sizes = params.hidden_layer_sizes
      self.activation = params.activation
      self.solver = params.solver
      self.alpha = params.alpha
      self.learning_rate = params.learning_rate
      self.max_iter = params.max_iter

      # Model starts as None
      self.model = None

  def __repr__(self):
      return f"""
      MLPRegressor(
          hidden_layer_sizes={self.hidden_layer_sizes},
          activation='{self.activation}',
          solver='{self.solver}',
          alpha={self.alpha},
          learning_rate='{self.learning_rate}',
          max_iter={self.max_iter}
      )
      """

  def fit(self, features, labels=None):
      assert self.model is None, 'A MODEL HAS ALREADY BEEN TRAINED'

      # Create and fit the MLP model
      mlp_model = MLPRegressor(
          hidden_layer_sizes=self.hidden_layer_sizes,
          activation=self.activation,
          solver=self.solver,
          alpha=self.alpha,
          learning_rate=self.learning_rate,
          max_iter=self.max_iter,
          random_state=42  # For reproducibility
      )

      # Fit the model
      mlp_model.fit(features, labels)

      # Save the trained model
      self.model = mlp_model

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
          "hidden_layer_sizes": (100, 50),
          "activation": "relu",
          "solver": "adam",
          "alpha": 0.001,
          "learning_rate": "constant",
          "max_iter": 300
      }
      model = custom_model(input_params)
      assert model.hidden_layer_sizes == (100, 50)
      assert model.activation == "relu"
      assert model.solver == "adam"
      assert model.alpha == 0.001
      assert model.learning_rate == "constant"
      assert model.max_iter == 300

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
          "hidden_layer_sizes": (100, 50),
          "activation": "relu",
          "solver": "adam",
          "alpha": 0.001,
          "learning_rate": "constant",
          "max_iter": 300
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