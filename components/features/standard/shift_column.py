from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import random, time

class shift_column_schema(base_schema):
    target_column: str
    shift_by: int = Field(ge=1)
    output_column: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, target_column: str, shift_by: int, output_column: str):
        params = shift_column_schema(target_column, shift_by, output_column)
        
        self.target_column = params.target_column
        self.shift_by = params.shift_by
        self.output_column = params.output_column
        
    def __repr__(self):
        return f'shift_column(target_column={self.target_column}, shift_by={self.shift_by})'

    def transform(self, dataframe: DataFrame):

        # MAKE SURE OUTPUT COLUMN IS UNIQUE
        existing_columns = list(dataframe.columns)
        exists_error = f"OUTPUT COLUMN '{self.output_column}' ALREADY EXISTS IN DATASET"
        assert self.output_column not in existing_columns, exists_error

        dataframe[self.output_column] = dataframe[self.target_column].shift(periods=self.shift_by)
        return dataframe
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        shift_column_schema(**self.yaml_params)

    def test_01_target_column_exists(self):
        target_column = self.yaml_params['target_column']

        # MAKE SURE A SAMPLE DATASET WAS PROVIDED BY THE PARENT PROCESS
        sample_error = f"UNITTEST ERROR: SAMPLE DATASET MISSING"
        self.assertTrue(hasattr(self, 'sample_dataset'), msg=sample_error)

        dataset: DataFrame = self.sample_dataset
        dataset_columns = list(dataset.columns)

        missing_error = f"TARGET COLUMN '{target_column}' MISSING FROM SAMPLE DATASET"
        self.assertIn(target_column, dataset_columns, msg=missing_error)

    def test_02_forward_shifting_works(self):
        dataset = DataFrame([{
            'symbol': 'SYNTH',
            'timestamp': int(time.time()) + x,
            'open': round(random.uniform(1, 3), 3),
            'close': round(random.uniform(1, 3), 3),
            'high': round(random.uniform(1, 3), 3),
            'low': round(random.uniform(1, 3), 3),
            'volume': random.randrange(50, 200),
        } for x in range(50)])

        # PICK A _POSITIVE_ NUMBER
        shift_by = random.randrange(5, 25)

        # APPLY THE FEATURE
        custom_feature(
            target_column='close',
            shift_by=shift_by,
            output_column='shifted_close',
        ).transform(dataset)

        # EXTRACT THE FEATURE VECTOR & AS WELL AS THE ORIGINAL COLUMN
        original_vector = dataset['close'].tolist()
        feature_vector = dataset['shifted_close'].tolist()

        # VERIFY THAT THE COLUMN WAS SHIFTED CORRECTLY
        self.assertEqual(original_vector[:-shift_by], feature_vector[shift_by:])