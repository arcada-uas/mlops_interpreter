from pydantic import BaseModel, Field
from components.features.base_feature import base_feature
from common.testing import base_unittest, validate_params
from pandas import DataFrame

class feature_input_schema(BaseModel):
    foo: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, foo: str):
        params = feature_input_schema(foo)
        self.foo = params.foo

    def __repr__(self):
        return f'my_feature(foo={self.foo})'

    def transform(self, dataframe: DataFrame):
        return dataframe

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_feature(**self.yaml_params)