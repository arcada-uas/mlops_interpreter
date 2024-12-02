from common.pydantic import base_schema
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.linear_model import LinearRegression

# model:
#     type: regression
#     name: sklearn_linear
#     params:
#         fit_intercept: True

class sklearn_linear_schema(base_schema):
    fit_intercept: bool

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, fit_intercept: bool):
        params = sklearn_linear_schema(fit_intercept)

        # SAVE PARAMS IN STATE
        self.fit_intercept = params.fit_intercept
        self.model = None

    def __repr__(self):
        return f'linear_regression(fit_intercept={self.fit_intercept}, prediction_window={self._prediction_window})'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(features, labels)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        sklearn_linear_schema(**self.yaml_params)