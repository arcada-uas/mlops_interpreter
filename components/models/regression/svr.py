from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.svm import SVR

# model:
#     type: regression
#     name: svr
#     params:
#         c_param: 1.0
#         epsilon: 0.1
#         kernel: rbf
#         degree: 3
#         gamma: scale
#         coef0: 0
#         shrinking: True
#         tol: 1e-3
#         max_iter: -1

class svg_regression_schema(base_schema):
    c_param: float = Field(gt=0, description="Regularization parameter for SVR.")
    epsilon: float = Field(ge=0, description="Epsilon-tube within which no penalty is associated in the training loss function.")
    kernel: str = Field(description="Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid').")
    degree: int = Field(ge=1, description="Degree of the polynomial kernel function ('poly'). Ignored by other kernels.")
    gamma: str = Field(description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.")
    coef0: float = Field(description="Independent term in kernel function ('poly' and 'sigmoid').")
    shrinking: bool = Field(description="Whether to use the shrinking heuristic.")
    tol: float = Field(gt=0, description="Tolerance for stopping criteria.")
    max_iter: int = Field(description="Maximum number of iterations (-1 for no limit).")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(
            self, c_param: float, epsilon: float, kernel: str, degree: int,
            gamma: str, coef0: float, shrinking: bool, tol: float, max_iter: int
        ):

        # VALIDATE INPUTS
        params = svg_regression_schema(
            c_param, epsilon, kernel, degree,
            gamma, coef0, shrinking, tol, max_iter
        )
        
        # SAVE MODEL PARAMS IN STATE
        self.c_param = params.c_param
        self.epsilon = params.epsilon
        self.kernel = params.kernel
        self.degree = params.degree
        self.gamma = params.gamma
        self.coef0 = params.coef0
        self.shrinking = params.shrinking
        self.tol = params.tol
        self.max_iter = params.max_iter

        # SHOULD ALWAYS DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f'svr_regression({
            self.stringify_vars([
                'c_param', 'epsilon', 'kernel',
                'degree', 'gamma', 'coef0',
                'shrinking', 'tol', 'max_iter',
            ])
        })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)

        # CREATE THE MODEL
        self.model = SVR(
            C=self.c_param,
            epsilon=self.epsilon,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            max_iter=self.max_iter
        )

        # TRAIN IT
        self.model.fit(features, labels)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):

        # CREATE THE MODELS
        model = custom_model(
            c_param=1,
            epsilon=0.1,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0,
            shrinking=True,
            tol=1e-3,
            max_iter=-1
        )

        # MAKE SURE PARAMS WERE SET CORRECTLY
        self.assertEqual(model.c_param, 1.0, "Mismatch in c_param.")
        self.assertEqual(model.epsilon, 0.1, "Mismatch in epsilon.")
        self.assertEqual(model.kernel, 'rbf', "Mismatch in kernel.")
        self.assertEqual(model.degree, 3, "Mismatch in degree.")
        self.assertEqual(model.gamma, 'scale', "Mismatch in gamma.")
        self.assertEqual(model.coef0, 0.0, "Mismatch in coef0.")
        self.assertEqual(model.shrinking, True, "Mismatch in shrinking.")
        self.assertEqual(model.tol, 1e-3, "Mismatch in tol.")
        self.assertEqual(model.max_iter, -1, "Mismatch in max_iter.")

    def test_01_runs_with_yaml_params(self):
        custom_model(**self.yaml_params)