from pydantic import BaseModel, Field
from components.features.base_feature import base_feature
from common.testing import base_unittest
from pandas import DataFrame
import random, time

class input_schema(BaseModel):
    foo: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, input_params: dict):

        # VALIDATE INPUT PARAMS
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
        params = input_schema(**input_params)

        # SAVE PARAMS IN STATE
        self.foo = params.foo

    def __repr__(self):
        return f"my_feature(foo={self.foo})"

    def transform(self, dataframe: DataFrame):
        # IMPLEMENT THE FEATURE
        # APPLY IT TO THE INPUT DATAFRAME
        return dataframe
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        pass