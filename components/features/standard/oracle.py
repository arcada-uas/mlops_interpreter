from components.features.base_feature import base_feature
from common.pydantic import base_schema, Field
from common.testing import base_unittest
from pandas import DataFrame
import numpy as np

class oracle_labeling_schema(base_schema):
    price_column: str = Field(min_length=1)
    transaction_fee: float = Field(ge=0.0)
    final_label: int = Field(ge=0, le=1)
    delta: float = Field(ge=0.0)
    output_column: str = Field(min_length=3, max_length=20)

##############################################################################################################
##############################################################################################################

class custom_feature(base_feature):
    def __init__(self, price_column: str, transaction_fee: float, final_label: int, delta: float, output_column: str):
        params = oracle_labeling_schema(
            price_column=price_column,
            transaction_fee=transaction_fee,
            final_label=final_label,
            delta=delta,
            output_column=output_column
        )

        self.price_column = params.price_column
        self.transaction_fee = params.transaction_fee
        self.final_label = params.final_label
        self.delta = params.delta
        self.output_column = params.output_column

    def __repr__(self):
        return (f'oracle_labeling(price_column={self.price_column}, '
                f'transaction_fee={self.transaction_fee}, '
                f'final_label={self.final_label}, delta={self.delta})')

    def transform(self, dataframe: DataFrame):
        # CHECK THAT PRICE COLUMN EXISTS
        assert self.price_column in dataframe.columns, f"PRICE COLUMN '{self.price_column}' MISSING FROM DATASET"

        # MAKE SURE OUTPUT COLUMN IS UNIQUE
        existing_columns = list(dataframe.columns)
        exists_error = f"OUTPUT COLUMN '{self.output_column}' ALREADY EXISTS IN DATASET"
        assert self.output_column not in existing_columns, exists_error

        # EXTRACT PRICES
        prices = dataframe[self.price_column].values

        # APPLY ORACLE LABELING FUNCTION
        labels = self.oracle_labeling(prices, self.transaction_fee, self.final_label, delta=self.delta)

        # ADD THE OUTPUT COLUMN
        dataframe[self.output_column] = labels

        return dataframe

    @staticmethod
    def oracle_labeling(prices, transaction_fee, final_label, delta=0):
        T = len(prices)
        S = np.full((T, 2), float('-inf'))  # Initialize cumulative log wealth
        backptr = np.zeros((T, 2), dtype=int)  # To reconstruct the path
        log_fee = np.log(1 - transaction_fee)

        # Initialize at t=0
        S[0, 0] = 0  # Log wealth when out of market
        S[0, 1] = log_fee  # If we enter the market at t=0

        # Forward pass: compute cumulative log wealth
        for t in range(1, T):
            for curr_state in [0, 1]:
                best_prev_S = float('-inf')
                best_prev_state = None
                for prev_state in [0, 1]:
                    S_prev = S[t - 1, prev_state]
                    r = None
                    # Compute log returns for each possible transition
                    if prev_state == 0 and curr_state == 0:
                        r = 0  # Stay out of market
                    elif prev_state == 0 and curr_state == 1:
                        r = log_fee  # Enter market (pay transaction fee)
                    elif prev_state == 1 and curr_state == 1:
                        price_change = prices[t] / prices[t - 1] - 1
                        if abs(price_change) >= delta:
                            r = np.log(1 + price_change)  # Hold position with significant price change
                        else:
                            r = 0  # Ignore small price changes
                    elif prev_state == 1 and curr_state == 0:
                        r = log_fee  # Exit market (pay transaction fee)
                    else:
                        continue
                    total_S = S_prev + r
                    if total_S > best_prev_S:
                        best_prev_S = total_S
                        best_prev_state = prev_state
                S[t, curr_state] = best_prev_S
                backptr[t, curr_state] = best_prev_state

        # Backward pass: reconstruct the optimal path
        y = np.zeros(T, dtype=int)
        y[-1] = final_label
        curr_state = final_label
        for t in range(T - 1, 0, -1):
            curr_state = backptr[t, curr_state]
            y[t - 1] = curr_state

        return y

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        oracle_labeling_schema(**self.yaml_params)

    def test_01_price_column_exists(self):
        price_column = self.yaml_params['price_column']

        # MAKE SURE A SAMPLE DATASET WAS PROVIDED BY THE PARENT PROCESS
        sample_error = f"UNITTEST ERROR: SAMPLE DATASET MISSING"
        self.assertTrue(hasattr(self, 'sample_dataset'), msg=sample_error)

        dataset: DataFrame = self.sample_dataset
        dataset_columns = list(dataset.columns)

        missing_error = f"PRICE COLUMN '{price_column}' MISSING FROM SAMPLE DATASET"
        self.assertIn(price_column, dataset_columns, msg=missing_error)

    def test_02_labels_correctness(self):
        # CREATE A SAMPLE DATASET
        np.random.seed(42)
        T = 50
        prices = np.cumprod(1 + np.random.randn(T) * 0.01) * 100  # Simulate a price series
        dataset = DataFrame({
            'timestamp': np.arange(T),
            'price': prices
        })

        # PARAMETERS
        transaction_fee = 0.0004  # 0.04%
        final_label = 1  # Desired final label
        delta = 0.001  # Ignore price changes smaller than 0.1%

        # APPLY THE FEATURE
        feature = custom_feature(
            price_column='price',
            transaction_fee=transaction_fee,
            final_label=final_label,
            delta=delta,
            output_column='labels'
        )
        transformed_dataset = feature.transform(dataset)

        # VERIFY THAT LABELS ARE 0 OR 1
        labels = transformed_dataset['labels']
        self.assertTrue(set(labels.unique()).issubset({0, 1}), msg="Labels should be 0 or 1")

        # CHECK THAT THE LENGTHS MATCH
        self.assertEqual(len(labels), len(prices), msg="Labels length does not match prices length")

        # OPTIONAL: ADD MORE DETAILED TESTS FOR LABEL CORRECTNESS