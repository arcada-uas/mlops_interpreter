from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest

class model_input_schema(base_schema):
    foo: str = Field(min_length=3)
    bar: int = Field(gt=5)

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, foo: str, bar: int):

        # VALIDATE INPUTS
        params = model_input_schema(foo, bar)

        # SAVE PARAMS IN STATE
        self.foo = params.foo
        self.bar = params.bar

        # MUST DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f'my_model({
            self.stringify_vars(['foo', 'bar'])
        })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(**self.yaml_params)