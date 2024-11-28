from pydantic import BaseModel
from components.models.base_model import base_model
from common.testing import base_unittest, validate_params
from sklearn.linear_model import LinearRegression

class input_schema(BaseModel):
    fit_intercept: bool

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)

        # SAVE PARAMS IN STATE
        self.fit_intercept = params.fit_intercept
        self.model = None

    def __repr__(self):
        return f'linear_regression(fit_intercept={self.fit_intercept})'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(features, labels)

    def predict(self, features: list[list[float]]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        
        return self.model.predict(features)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)