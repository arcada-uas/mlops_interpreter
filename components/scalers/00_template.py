from components.features.base_feature import base_feature
from common.testing import base_unittest, validate_params
from pydantic import BaseModel
from pandas import DataFrame

class input_schema(BaseModel):
    foo: bool

##############################################################################################################
##############################################################################################################

class custom_scaler(base_feature):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)
        self.foo = params.foo

        # SHOULD ALWAYS DEFAULT TO NONE
        self.scaler = None

    def __repr__(self):
        return f"scaler(foo={self.foo})"

    # SKLEARN REQUIRED METHOD -- FITS SCALER, THEN TRANSFORMS
    def fit_transform(self, features: DataFrame, labels=None):
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"

        self.fit(features)
        return self.transform(features)

    # FIT THE SCALER ON TRAINING DATA
    def fit(self, dataframe: DataFrame, labels=None):
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"
        pass

    # TRANSFORM BASED ON FITTING
    def transform(self, dataframe: DataFrame):
        assert self.scaler != None, f"THE SCALER HAS _NOT_ BEEN FIT YET"
        pass

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_params(self):
        custom_scaler(self.input_params)