from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import random, time

class extract_features_schema(base_schema):
    columns: list[str] = Field(min_length=1)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, columns: list[str]):
        params = extract_features_schema(columns)
        self.columns = params.columns
        
    def __repr__(self):
        return f'extract_features({ ', '.join(self.columns) })'
    
    def transform(self, dataframe: DataFrame):
        df_columns = list(dataframe.columns)

        # MAKE SURE ALL THE COLUMNS EXIST
        for column_name in self.columns:
            assert column_name in df_columns, f"COLUMN '{column_name}' MISSING FROM DATASET"

        # EXTRACT JUST THE VALUES OF THE FEATURE COLUMNS
        return dataframe[self.columns]

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_schema(self):
        extract_features_schema(**self.yaml_params)

    def test_01_demo(self):
        dataset_length = random.randrange(25, 50)

        # GENERATE A SYNTHETIC DATASET
        dataset = [{
            'symbol': 'SYNTH',
            'timestamp': int(time.time())  + x,
            'open': round(random.uniform(1, 3), 3),
            'close': round(random.uniform(1, 3), 3),
            'volume': random.randrange(50, 200),
        } for x in range(dataset_length)]

        # SELECT JUST THE OPEN & CLOSE COLUMNS
        expected_output = [{
            'open': x['open'],
            'close': x['close']
        } for x in dataset]

        # CONVERT TO DATAFRAME & PASS IT THROUGH THE FEATURE
        dataset_df = DataFrame(dataset)
        dataset_df = custom_feature(columns=['open', 'close']).transform(dataset_df)

        # MAKE SURE OUTPUT CONTENTS MATCH EXPECTATION
        self.assertEqual(dataset_df.to_dict(orient='records'), expected_output)