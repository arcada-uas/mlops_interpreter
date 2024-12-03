from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import random, time


class quartile_label_schema(base_schema):
    target_column: str
    output_column: str = Field(min_length=3, max_length=20)


##############################################################################################################
##############################################################################################################


class custom_feature(base_feature):
    def __init__(self, target_column: str, output_column: str):
        params = quartile_label_schema(target_column=target_column, output_column=output_column)
        self.target_column = params.target_column
        self.output_column = params.output_column

    def __repr__(self):
        return f'quartile_label(target_column={self.target_column})'

    def transform(self, dataframe: DataFrame):
        # MAKE SURE OUTPUT COLUMN IS UNIQUE
        existing_columns = list(dataframe.columns)
        exists_error = f"OUTPUT COLUMN '{self.output_column}' ALREADY EXISTS IN DATASET"
        assert self.output_column not in existing_columns, exists_error

        # COMPUTE PERCENTILE VALUES
        p20 = dataframe[self.target_column].quantile(0.20)
        p45 = dataframe[self.target_column].quantile(0.45)
        p55 = dataframe[self.target_column].quantile(0.55)
        p80 = dataframe[self.target_column].quantile(0.80)

        # APPLY QUARTILE LABELING LOGIC
        def assign_label(x):
            if x <= p20:
                return 0  # Represents the lowest values, potentially "sell"
            elif x <= p45:
                return 1  # Represents below-average values
            elif x <= p55:
                return 2  # Represents little change or "hold"
            elif x <= p80:
                return 3  # Represents above-average values
            else:
                return 4  # Represents the highest values, potentially "buy"

        dataframe[self.output_column] = dataframe[self.target_column].apply(assign_label)
        return dataframe


##############################################################################################################
##############################################################################################################


class tests(base_unittest):
    def test_00_validate_input(self):
        quartile_label_schema(**self.yaml_params)

    def test_01_target_column_exists(self):
        target_column = self.yaml_params['target_column']
        sample_error = f"UNITTEST ERROR: SAMPLE DATASET MISSING"
        self.assertTrue(hasattr(self, 'sample_dataset'), msg=sample_error)

        dataset: DataFrame = self.sample_dataset
        dataset_columns = list(dataset.columns)
        missing_error = f"TARGET COLUMN '{target_column}' MISSING FROM SAMPLE DATASET"
        self.assertIn(target_column, dataset_columns, msg=missing_error)

    def test_02_quartile_logic_works(self):
        dataset = DataFrame([{
            'symbol': 'SYNTH',
            'timestamp': int(time.time()) + x,
            'open': round(random.uniform(1, 3), 3),
            'close': round(random.uniform(1, 3), 3),
            'high': round(random.uniform(1, 3), 3),
            'low': round(random.uniform(1, 3), 3),
            'volume': random.randrange(50, 200),
        } for x in range(1000)])

        # APPLY THE FEATURE
        feature = custom_feature(
            target_column='close',
            output_column='quartile_label'
        )
        transformed_dataset = feature.transform(dataset)

        # COMPUTE PERCENTILE VALUES
        p20 = dataset['close'].quantile(0.20)
        p45 = dataset['close'].quantile(0.45)
        p55 = dataset['close'].quantile(0.55)
        p80 = dataset['close'].quantile(0.80)

        # VERIFY THE LOGIC
        def expected_label(x):
            if x <= p20:
                return 0
            elif x <= p45:
                return 1
            elif x <= p55:
                return 2
            elif x <= p80:
                return 3
            else:
                return 4

        expected_labels = dataset['close'].apply(expected_label)
        computed_labels = transformed_dataset['quartile_label']
        self.assertTrue(
            (expected_labels == computed_labels).all(),
            msg="Quartile labeling logic failed"
        )
