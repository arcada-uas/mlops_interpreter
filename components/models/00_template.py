from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.metrics import r2_score

class input_schema(BaseModel):
    alpha: float = Field(ge=0)
    max_iter: int = Field(ge=5)

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):

        # VALIDATE INPUT PARAMS
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
        params = input_schema(**input_params)

        # SAVE PARAMS IN STATE
        self.alpha = params.alpha
        self.max_iter = params.max_iter

        # MAKE SURE MODEL DEFAULTS TO NONE
        self.model = None

    def __repr__(self):
        return f"xgboost(alpha={self.alpha}, max_iter={self.max_iter})"

    def fit(self, features: list[list[float]], labels: list[float] = None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'

        # TRAIN THE MODEL
        model = 'foo'

        # SAVE THE TRAINED MODEL IN THE STATE
        self.model = model

    def predict(self, features: list[list[float]]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)
    
    def score(self, features: list[list[float]], labels: list[float]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'

        # MAKE PREDICTIONS & COMPUTE R2 SCORE
        predictions = self.model.predict(features)
        return r2_score(labels, predictions)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)