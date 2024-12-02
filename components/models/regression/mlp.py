from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.neural_network import MLPRegressor

# model:
#     type: regression
#     name: mlp
#     params:
#         layer_sizes: 
#             - 200
#             - 100
#             - 50
#         activation_func: relu
#         solver_func: adam
#         alpha: 0.001
#         learning_rate: adaptive
#         max_iter: 1000
#         early_stopping: True
#         validation_fraction: 0.1
#         tolerance: 1e-4

class mlp_regression_schema(base_schema):
    layer_sizes: list[int] = Field(description="List of number of neurons in each hidden layer.")
    activation_func: str = Field(description="Activation function for the hidden layers.")
    solver_func: str = Field(description="Optimization solver.")
    alpha: float = Field(gt=0, description="Regularization term (L2 penalty).")
    learning_rate: str = Field(description="Learning rate schedule for optimization.")
    max_iter: int = Field(ge=1, description="Maximum number of iterations.")
    early_stopping: bool = Field(description="Enable early stopping based on validation.")
    tolerance: float = Field(gt=0, description="Improvement tolerance until early-stopping.")
    validation_fraction: float = Field(gt=0, le=1, description="Proportion of validation set during training.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(
            self, layer_sizes: tuple, activation_func: str, solver_func: str, alpha: float,
            learning_rate: str, max_iter: int, early_stopping: bool, tolerance: float, validation_fraction: float
        ):

        # VALIDATE PARAMS
        params = mlp_regression_schema(
            layer_sizes, activation_func, solver_func, alpha, 
            learning_rate, max_iter, early_stopping,
            tolerance, validation_fraction
        )

        # SAVE MODEL PARAMS IN STATE
        self.layer_sizes = params.layer_sizes
        self.activation_func = params.activation_func
        self.solver_func = params.solver_func
        self.alpha = params.alpha
        self.learning_rate = params.learning_rate
        self.max_iter = params.max_iter
        self.early_stopping = params.early_stopping
        self.tolerance = params.tolerance
        self.validation_fraction = params.validation_fraction

        # SHOULD ALWAYS DEFAULT TO FALSE
        self.model = None

    def __repr__(self):
        return f'mlp_regression({
            self.stringify_vars([
                'layer_sizes', 'activation_func', 'solver_func', 'alpha',
                'learning_rate', 'max_iter', 'early_stopping',
                'tolerance', 'validation_fraction'
            ])
        })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)

        # CREATE THE MODEL
        self.model = MLPRegressor(
            hidden_layer_sizes=self.layer_sizes,
            activation=self.activation_func,
            solver=self.solver_func,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            tol=self.tolerance,
            validation_fraction=self.validation_fraction,
            random_state=42,
            verbose=True
        )

        # TRAIN IT
        self.model.fit(features, labels)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):

        # CREATE THE MODELS
        model = custom_model(
            layer_sizes=[200, 100, 50],
            activation_func='relu',
            solver_func='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            tolerance=1e-4,
            validation_fraction=0.1,
        )

        # MAKE SURE PARAMS WERE SET CORRECTLY
        self.assertEqual(model.layer_sizes, [200, 100, 50], "Mismatch in hidden_layer_sizes")
        self.assertEqual(model.activation_func, "relu", "Mismatch in activation_func")
        self.assertEqual(model.solver_func, "adam", "Mismatch in solver_func")
        self.assertEqual(model.alpha, 0.001, "Mismatch in alpha")
        self.assertEqual(model.learning_rate, "adaptive", "Mismatch in learning_rate")
        self.assertEqual(model.max_iter, 1000, "Mismatch in max_iter")
        self.assertTrue(model.early_stopping, "Mismatch in early_stopping")
        self.assertEqual(model.validation_fraction, 0.1, "Mismatch in validation_fraction")
        self.assertEqual(model.tolerance, 1e-4, "Mismatch in validation_fraction")

    def test_01_runs_with_yaml_params(self):
        custom_model(**self.yaml_params)