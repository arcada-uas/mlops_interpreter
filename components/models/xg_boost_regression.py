from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest, validate_params
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# class cv_schema(BaseModel):
#     min: float = Field(gt=0)
#     max: float = Field(gt=0)

class input_schema(BaseModel):
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0)
    # cross_validation: cv_schema

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

        # self.cv_min = params.cross_validation.min
        # self.cv_max = params.cross_validation.max

        # MAKE SURE MODEL DEFAULTS TO NONE
        self.model = None

    def __repr__(self):
        return f"xg_boost_regression(n_estimators={self.n_estimators}, max_depth={self.max_depth}, learning_rate={self.learning_rate}, subsample={self.subsample})"

    def fit(self, features: list[list[float]], labels: list[float] = None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'

        # ATTACH PARAMS TO MODEL
        xgboost_model = XGBRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            learning_rate=self.learning_rate, 
            subsample=self.subsample,
            n_jobs=-1
        )

        # TRAIN & SAVE IT TO STATE
        xgboost_model.fit(features, labels)
        self.model = xgboost_model

    def predict(self, features: list[list[float]]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)

    def score(self, features: list[list[float]], labels: list[float]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'

        # MAKE PREDICTIONS ON GIVEN FEATURES
        predictions = self.model.predict(features)

        # COMPUTE ACCURACY
        return {
            'r2': round(r2_score(labels, predictions), 4),
            'mse': round(float(mean_squared_error(labels, predictions)), 4)
        }

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)