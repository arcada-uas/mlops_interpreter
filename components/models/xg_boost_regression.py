from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest, validate_params
from xgboost import XGBRegressor

class input_schema(BaseModel):
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0)

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)

        # SAVE PARAMS IN STATE
        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample

        # DEFAULT PARAMSx
        self.model = None

    def __repr__(self):
        param_names = ['n_estimators', 'max_depth', 'learning_rate', 'subsample']
        param_string = ', '.join([f'{x}={self.__dict__[x]}' for x in param_names])

        return f'xg_boost_regression({ param_string })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'

        # INSTANTIATE THE MODEL
        self.model = XGBRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            learning_rate=self.learning_rate, 
            subsample=self.subsample,
            n_jobs=-1
        )

        # THEN TRAIN IT
        self.model.fit(features, labels)

    def predict(self, features: list[list[float]]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)