from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import numpy as np
import random, time

class log_return_schema(base_schema):
    target_column: str
    output_column: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, target_column: str, output_column: str):
        params = log_return_schema(target_column=target_column, output_column=output_column)
        
        self.target_column = params.target_column
        self.output_column = params.output_column
        
    def __repr__(self):
        return f'log_return(target_column={self.target_column})'

    def transform(self, dataframe: DataFrame):

        # MAKE SURE OUTPUT COLUMN IS UNIQUE
        existing_columns = list(dataframe.columns)
        exists_error = f"OUTPUT COLUMN '{self.output_column}' ALREADY EXISTS IN DATASET"
        assert self.output_column not in existing_columns, exists_error

        # APPLY LOG RETURN LOGIC
        dataframe[self.output_column] = np.log(dataframe[self.target_column]).diff()

        return dataframe
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        log_return_schema(**self.yaml_params)

    def test_01_target_column_exists(self):
        target_column = self.yaml_params['target_column']

        # MAKE SURE A SAMPLE DATASET WAS PROVIDED BY THE PARENT PROCESS
        sample_error = f"UNITTEST ERROR: SAMPLE DATASET MISSING"
        self.assertTrue(hasattr(self, 'sample_dataset'), msg=sample_error)

        dataset: DataFrame = self.sample_dataset
        dataset_columns = list(dataset.columns)

        missing_error = f"TARGET COLUMN '{target_column}' MISSING FROM SAMPLE DATASET"
        self.assertIn(target_column, dataset_columns, msg=missing_error)

    # def test_02_log_return_logic_works(self):
    #     dataset = DataFrame([{
    #         'symbol': 'SYNTH',
    #         'timestamp': int(time.time()) + x,
    #         'open': round(random.uniform(1.5, 3.0), 3),
    #         'close': round(random.uniform(1.5, 3.0), 3),
    #         'high': round(random.uniform(1.5, 3.0), 3),
    #         'low': round(random.uniform(1.5, 3.0), 3),
    #         'volume': random.randrange(50, 200),
    #     } for x in range(50)])

    #     feature = custom_feature(
    #         target_column='close',
    #         output_column='log_return'
    #     )
    #     transformed_dataset = feature.transform(dataset)

    #     log_close = np.log(dataset['close'])
    #     expected_log_return = log_close.diff().fillna(0)
    #     computed_vector = transformed_dataset['log_return']

    #     is_close = np.allclose(computed_vector, expected_log_return, atol=1e-8)

    #     if not is_close:
    #         for i, (expected, computed) in enumerate(zip(expected_log_return, computed_vector)):
    #             if not np.isclose(expected, computed, atol=1e-8):
    #                 print(f"Mismatch at index {i}: Expected {expected}, Computed {computed}")

    #     self.assertTrue(is_close, "Log return calculation does not match expected values.")

