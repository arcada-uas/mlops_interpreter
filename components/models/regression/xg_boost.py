from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from xgboost import XGBRegressor

# model:
#     type: regression
#     name: xg_boost
#     params:
#         n_estimators: 100
#         max_depth: 6
#         learning_rate: 0.3
#         subsample: 1.0
#         colsample_bytree: 1.0
#         gamma: 0.0
#         reg_alpha: 0.0
#         reg_lambda: 1.0

class xg_boost_schema(base_schema):
    n_estimators: int = Field(ge=1, description="Number of trees in the XGBoost model.")
    max_depth: int = Field(ge=1, description="Maximum depth of the trees.")
    learning_rate: float = Field(gt=0, description="Learning rate for the model.")
    subsample: float = Field(gt=0, le=1, description="Subsample ratio of the training data.")
    colsample_bytree: float = Field(gt=0, le=1, description="Subsample ratio of columns when constructing each tree.")
    gamma: float = Field(ge=0, description="Minimum loss reduction required to make a further partition on a leaf node.")
    reg_alpha: float = Field(ge=0, description="L1 regularization term on weights.")
    reg_lambda: float = Field(ge=0, description="L2 regularization term on weights.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(
            self, n_estimators: int, max_depth: int, learning_rate: float, subsample: float,
            colsample_bytree: float, gamma: float, reg_alpha: float, reg_lambda: float
        ):

        # VALIDATE INPUTS
        params = xg_boost_schema(
            n_estimators, max_depth, learning_rate, subsample,
            colsample_bytree, gamma, reg_alpha, reg_lambda
        )

        # SAVE PARAMS IN STATE
        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample
        self.colsample_bytree = params.colsample_bytree
        self.gamma = params.gamma
        self.reg_alpha = params.reg_alpha
        self.reg_lambda = params.reg_lambda

        # SHOULD DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f'xg_boost_regression({
            self.stringify_vars([
                'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
            ])
        })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)

        # INSTANTIATE THE MODEL
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=-1,
            random_state=42
        )

        # THEN TRAIN IT
        self.model.fit(features, labels)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):

        # CREATE THE MODELS
        model = custom_model(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1
        )

        # MAKE SURE PARAMS WERE SET CORRECTLY
        self.assertEqual(model.n_estimators, 100, "Mismatch in n_estimators.")
        self.assertEqual(model.max_depth, 6, "Mismatch in max_depth.")
        self.assertEqual(model.learning_rate, 0.3, "Mismatch in learning_rate.")
        self.assertEqual(model.subsample, 0.8, "Mismatch in subsample.")
        self.assertEqual(model.colsample_bytree, 0.8, "Mismatch in colsample_bytree.")
        self.assertEqual(model.gamma, 0.0, "Mismatch in gamma.")
        self.assertEqual(model.reg_alpha, 0.0, "Mismatch in reg_alpha.")
        self.assertEqual(model.reg_lambda, 1.0, "Mismatch in reg_lambda.")

    def test_01_runs_with_yaml_params(self):
        custom_model(**self.yaml_params)