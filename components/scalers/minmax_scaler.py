from components.scalers.base_scaler import base_scaler
from common.testing import base_unittest, create_synth_dataset
from sklearn.preprocessing import MinMaxScaler
from common.pydantic import base_schema
from pandas import DataFrame

# scaler:
#     name: minmax_scaler
#     params:
#         min_range: 0
#         max_range: 1

class minmax_scaler_schema(base_schema):
    min_range: int
    max_range: int

##############################################################################################################
##############################################################################################################

class custom_scaler(base_scaler):
    def __init__(self, min_range: int, max_range: int):
        params = minmax_scaler_schema(min_range=min_range, max_range=max_range)

        # MAKE SURE MIN VALUE IS SMALLER
        assert params.min_range < params.max_range, f"MIN VALUE MUST BE SMALLER THAN MAX VALUE"

        # SAVE PARAMS IN STATE
        self.min = params.min_range
        self.max = params.max_range

        # DEFAULT PARAMS
        self.scaler = None

    def __repr__(self):
        return f'minmax_scaler(min={self.min}, max={self.max})'

    # FIT THE SCALER ON TRAINING DATA
    def fit(self, features: DataFrame, labels=None):
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"
        self.scaler = MinMaxScaler(feature_range=(self.min, self.max))
        self.scaler.fit(features)

    # TRANSFORM BASED ON FITTING
    def transform(self, features: DataFrame):
        assert self.scaler != None, f"THE SCALER HAS _NOT_ BEEN FIT YET"
        return self.scaler.transform(features)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_params(self):
        params = minmax_scaler_schema(**self.yaml_params)
        self.assertGreater(params.max_range, params.min_range, 'MIN VALUE CANNOT BE LARGER THAN MAX VALUE')

    def test_01_scaling_range_works_correctly(self):

        # GENERATE A SYNTH DATASET
        dataset = create_synth_dataset(
            column_names=['open', 'close'],
            random_state=147,
            num_rows=5,
            columns_only=True
        )

        # CREATE THE SCALER
        scaler = custom_scaler(min_range=-69, max_range=420)

        # FIT & TRANSFORM FEATURES
        scaler.fit(dataset)
        scaled_features = scaler.transform(dataset).tolist()

        # STITCH TOGETHER EXPECTED OUTPUT
        expected_output = [
            [-69.0, 228.85770723142053], 
            [178.66476939453023, 28.745879024649277], 
            [110.40483913994427, 420.0], 
            [191.49127940641083, -69.0], 
            [420.0, 252.79071360340453]
        ]

        # MAKE SURE OUTPUT MATCHES EXPECTATION
        self.assertEqual(scaled_features, expected_output, 'MINMAX SCALER OUTPUT DOES NOT MATCH')
