from components.features.base_feature import base_feature
from common.testing import base_unittest, validate_params
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from pandas import DataFrame

class input_schema(BaseModel):
    use_std: bool
    use_mean: bool
    matrix_conversion: bool

##############################################################################################################
##############################################################################################################

class custom_scaler(base_feature):
    def __init__(self, input_params: dict):
        params = validate_params(input_params, input_schema)
        self.mean = params.use_mean
        self.std = params.use_std
        self.convert = params.matrix_conversion

        # SHOULD ALWAYS DEFAULT TO NONE
        self.scaler = None

    def __repr__(self):
        return f"scaler_pipeline(matrix_conversion={self.convert}, use_mean={self.mean}, use_std={self.std})"
    
    # CONVERT DATAFRAME TO FEATURE MATRIX
    def df_to_matrix(self, dataframe):
        return dataframe.values.tolist()

    # SKLEARN REQUIRED METHOD -- FITS SCALER, THEN TRANSFORMS
    def fit_transform(self, features: DataFrame, labels=None):
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"

        self.fit(features)
        return self.transform(features)

    # FIT THE SCALER ON TRAINING DATA
    def fit(self, features: DataFrame, labels=None):
        assert isinstance(features, DataFrame), f"FEATURES SHOULD BE OF TYPE DATAFRAME, GOT {type(features)}"
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"

        # INSTANTIATE THE SCALER
        self.scaler = StandardScaler(
            with_std=self.std, 
            with_mean=self.mean
        )

        # WHEN REQUESTED, CONVERT DF TO FLOAT MATRIX
        if self.convert:
            features: list[list[float]] = self.df_to_matrix(features)

        self.scaler.fit(features)

    # TRANSFORM BASED ON FITTING
    def transform(self, features: DataFrame):
        assert isinstance(features, DataFrame), f"FEATURES SHOULD BE OF TYPE DATAFRAME, GOT {type(features)}"
        assert self.scaler != None, f"THE SCALER HAS _NOT_ BEEN FIT YET"

        # WHEN REQUESTED, CONVERT DF TO FLOAT MATRIX
        if self.convert:
            features: list[list[float]] = self.df_to_matrix(features)

        return self.scaler.transform(features)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_params(self):
        custom_scaler(self.input_params)

    def test_01_df_to_feature_matrix(self):
        dataset = DataFrame([{
            'open': 6.674 + x,
            'close': 2.452 + x,
        } for x in range(10)])

        # GENERATE THE FEATURE MATRIX
        # OF THE OPEN & CLOSE COLUMNS
        real_output = custom_scaler({
            'use_std': True,
            'use_mean': True,
            'matrix_conversion': True
        }).df_to_matrix(dataset)

        # WHAT WE EXPECT TO RECEIVE
        expected_output = [
            [6.674, 2.452], [7.674, 3.452], [8.674, 4.452], [9.674, 5.452], 
            [10.674, 6.452], [11.674, 7.452], [12.674, 8.452], [13.674, 9.452], 
            [14.674, 10.452], [15.674, 11.452]
        ]

        # MAKE SURE THEY MATCH
        output_error = f"OUTPUT MATRIX DOES NOT MATCH EXPECTED VALUE"
        self.assertEqual(real_output, expected_output, msg=output_error)