from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest, validate_params

class input_schema(BaseModel):
    foo: float = Field(ge=0)

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)
        self.foo = params.foo

        # MAKE SURE MODEL DEFAULTS TO NONE
        self.model = None

    def __repr__(self):
        return f"my_model(foo={self.foo})"

    def fit(self, features: list[list[float]], labels: list[float] = None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'
        pass

    def predict(self, features: list[list[float]]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        pass
    
    def score(self, features: list[list[float]], labels: list[float]):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        pass
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)