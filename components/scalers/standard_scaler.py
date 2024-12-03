from components.scalers.base_scaler import base_scaler
from common.testing import base_unittest, create_synth_dataset
from sklearn.preprocessing import StandardScaler
from common.pydantic import base_schema
from pandas import DataFrame

# scaler:
#     name: standard_scaler
#     params:
#         use_std: True
#         use_mean: True

class standard_scaler_schema(base_schema):
    use_std: bool
    use_mean: bool

##############################################################################################################
##############################################################################################################

class custom_scaler(base_scaler):
    def __init__(self, use_std: bool, use_mean: bool):
        params = standard_scaler_schema(use_std, use_mean)

        # MAKE SURE AT LEAST ONE FUNC WAS SELECTED
        missing_func_error = "ARGS 'use_mean' AND 'use_std' CANNOT BOTH BE FALSE, SELECT EITHER OR BOTH."
        assert params.use_mean or params.use_std, missing_func_error

        # SAVE PARAMS IN STATE
        self.mean = params.use_mean
        self.std = params.use_std

        # DEFAULT PARAMS
        self.scaler = None

    def __repr__(self):
        return f'standard_scaler(std={self.std}, mean={self.mean})'

    # FIT THE SCALER ON TRAINING DATA
    def fit(self, features: DataFrame, labels=None):
        assert self.scaler == None, f"THE SCALER HAS ALREADY BEEN FIT"

        self.scaler = StandardScaler(with_std=self.std, with_mean=self.mean)
        self.scaler.fit(features)

    # TRANSFORM BASED ON FITTING
    def transform(self, features: DataFrame):
        assert self.scaler != None, f"THE SCALER HAS _NOT_ BEEN FIT YET"
        return self.scaler.transform(features)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_params(self):
        params = standard_scaler_schema(**self.yaml_params)

        # MAKE SURE AT LEAST ONE FUNC WAS SELECTED
        missing_func_error = "ARGS 'use_mean' AND 'use_std' CANNOT BOTH BE FALSE, SELECT EITHER OR BOTH."
        self.assertTrue(params.use_mean or params.use_std, missing_func_error)

    def test_01_scaling_with_both(self):

        # GENERATE A SYNTH DATASET
        dataset = create_synth_dataset(
            column_names=['open', 'close'],
            random_state=142,
            num_rows=5,
            columns_only=True
        )

        # CREATE THE SCALER
        scaler = custom_scaler(use_mean=True, use_std=True)

        # FIT & TRANSFORM FEATURES
        scaler.fit(dataset)
        scaled_features = scaler.transform(dataset).tolist()

        # STITCH TOGETHER EXPECTED OUTPUT
        expected_output = [
            [1.3866163702828662, -1.2733360696751619], 
            [0.9587497008977858, 0.07995161135932118], 
            [-0.9825681570175887, -0.9166676847647452], 
            [-1.0447478198040108, 1.446039706822884], 
            [-0.31805009435905224, 0.6640124362577018]
        ]

        # MAKE SURE OUTPUT MATCHES EXPECTATION
        self.assertEqual(scaled_features, expected_output, 'STANDARD SCALER OUTPUT DOES NOT MATCH')

    def test_02_scaling_with_mean(self):

        # GENERATE A SYNTH DATASET
        dataset = create_synth_dataset(
            column_names=['open', 'close'],
            random_state=143,
            num_rows=5,
            columns_only=True
        )

        # CREATE THE SCALER
        scaler = custom_scaler(use_mean=True, use_std=False)

        # FIT & TRANSFORM FEATURES
        scaler.fit(dataset)
        scaled_features = scaler.transform(dataset).tolist()

        # STITCH TOGETHER EXPECTED OUTPUT
        expected_output = [
            [0.35937723797662136, -0.3619568655431068], 
            [-0.22216861450105452, 1.3443811231559053], 
            [0.1219061139346832, -0.3353985036123584], 
            [1.6056305744683863, 0.2849150377439952], 
            [-1.8647453118786366, -0.9319407917444353]
        ]

        # MAKE SURE OUTPUT MATCHES EXPECTATION
        self.assertEqual(scaled_features, expected_output, 'STANDARD SCALER OUTPUT DOES NOT MATCH')

    def test_03_scaling_with_std(self):

        # GENERATE A SYNTH DATASET
        dataset = create_synth_dataset(
            column_names=['open', 'close'],
            random_state=144,
            num_rows=5,
            columns_only=True
        )

        # CREATE THE SCALER
        scaler = custom_scaler(use_mean=False, use_std=True)

        # FIT & TRANSFORM FEATURES
        scaler.fit(dataset)
        scaled_features = scaler.transform(dataset).tolist()

        # STITCH TOGETHER EXPECTED OUTPUT
        expected_output = [
            [2.390123627730282, 0.07049750998631868], 
            [1.3244576722330508, -1.1770897061612404], 
            [-0.2352427896485497, 1.378535150911083], 
            [-0.11923077178809321, -1.3064137092228658], 
            [0.24196049263675878, -0.8395075429229854]
        ]

        # MAKE SURE OUTPUT MATCHES EXPECTATION
        self.assertEqual(scaled_features, expected_output, 'STANDARD SCALER OUTPUT DOES NOT MATCH')
