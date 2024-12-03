from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import random, time

class buy_sell_schema(base_schema):
    target_column: str
    threshold: float = Field(ge=0.0)
    output_column: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, target_column: str, threshold: float, output_column: str):
        params = buy_sell_schema(target_column=target_column, threshold=threshold, output_column=output_column)
        
        self.target_column = params.target_column
        self.threshold = params.threshold
        self.output_column = params.output_column
        
    def __repr__(self):
        return f'buy_sell(target_column={self.target_column}, threshold={self.threshold})'

    def transform(self, dataframe: DataFrame):

        # MAKE SURE OUTPUT COLUMN IS UNIQUE
        existing_columns = list(dataframe.columns)
        exists_error = f"OUTPUT COLUMN '{self.output_column}' ALREADY EXISTS IN DATASET"
        assert self.output_column not in existing_columns, exists_error

        # APPLY BUY/SELL LOGIC
        dataframe[self.output_column] = (dataframe[self.target_column].diff() > self.threshold).astype(int)

        return dataframe
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        buy_sell_schema(**self.yaml_params)

    def test_01_target_column_exists(self):
        target_column = self.yaml_params['target_column']

        # MAKE SURE A SAMPLE DATASET WAS PROVIDED BY THE PARENT PROCESS
        sample_error = f"UNITTEST ERROR: SAMPLE DATASET MISSING"
        self.assertTrue(hasattr(self, 'sample_dataset'), msg=sample_error)

        dataset: DataFrame = self.sample_dataset
        dataset_columns = list(dataset.columns)

        missing_error = f"TARGET COLUMN '{target_column}' MISSING FROM SAMPLE DATASET"
        self.assertIn(target_column, dataset_columns, msg=missing_error)

    def test_02_threshold_logic_works(self):
        dataset = DataFrame([{
            'symbol': 'SYNTH',
            'timestamp': int(time.time()) + x,
            'open': round(random.uniform(1, 3), 3),
            'close': round(random.uniform(1, 3), 3),
            'high': round(random.uniform(1, 3), 3),
            'low': round(random.uniform(1, 3), 3),
            'volume': random.randrange(50, 200),
        } for x in range(50)])

        # PICK A THRESHOLD
        threshold = 0.00005

        # APPLY THE FEATURE
        feature = custom_feature(
            target_column='close',
            threshold=threshold,
            output_column='buy_sell'
        )
        transformed_dataset = feature.transform(dataset)

        # VERIFY THE LOGIC
        diff_vector = dataset['close'].diff() > threshold
        computed_vector = transformed_dataset['buy_sell'] == diff_vector.astype(int)
        self.assertTrue(computed_vector.all())
