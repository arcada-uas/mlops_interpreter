## OVERVIEW
todo

## WORKFLOW
1. Implement your functionality in `components/`:
    - [Data retrieval methods](components/data_retrieval/)
    - [Segmentation methods](components/segmentation/)
    - [Features](components/features/)
    - [Models](components/models/)
    - [Scalers](components/scalers/)
    - [Trading strategies](components/trading_strategies/)
2. Make your functionality available to the interpreter in `mappings/`.
    - [Data retrieval methods](mappings/data_retrieval.py)
    - [Segmentation methods](mappings/segmentation.py)
    - [Features](mappings/features.py)
    - [Models](mappings/models.py)
    - [Scalers](mappings/scalers.py)
3. Reference your functionality in the YAML configuration file.
    - Default: [`pipeline.yaml`](pipeline.yaml)
    - More examples in `experiments/`
4. Automatically unittest the pipeline composition using your yaml config.
    - `make test`
    - If some tests fail, fix the yaml parameter based on the error message.
5. Build the pipeline using your yaml config.
    - `make run`

## SCRIPTS
- Tested with `make` version `v4.4.1`
```sh
# INSTALL PIP DEPENDENCIES 
make install
```

```sh
# RUN UNITTESTS
make test

# EXAMPLE OUTPUT
test_00_input_schema (components.data_retrieval.from_cassandra.tests.test_00_input_schema) ... ok
test_01_timestamp_format (components.data_retrieval.from_cassandra.tests.test_01_timestamp_format) ... ok
test_02_timestamp_order (components.data_retrieval.from_cassandra.tests.test_02_timestamp_order) ... ok
test_03_cassandra_connection (components.data_retrieval.from_cassandra.tests.test_03_cassandra_connection) ... ok
test_04_ascending_order (components.data_retrieval.from_cassandra.tests.test_04_ascending_order) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.174s

OK
test_00_validate_input (components.features.stochastic_k.tests.test_00_validate_input) ... ok
test_01_demo (components.features.stochastic_k.tests.test_01_demo) ... ok
test_02_required_columns (components.features.stochastic_k.tests.test_02_required_columns) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.003s
OK
```

```sh
# CREATE SKLEARN PIPELINE
make run

# EXAMPLE OUTPUT
--------
DATASET LEN:        8605
TRAIN SEGMENT:      6454
TEST SEGMENT:       1291
VALIDATE SEGMENT:   860
--------
TRAIN SCORE:        0.9836
TEST SCORE:         0.7872
VALIDATE SCORE:     0.8906
--------
Pipeline(steps=[
    ('hidden_to_df', to_dataframe()),
    ('0_stochastic_k', stochastic_k(window_size=5)),
    ('1_shift_column', shift_column(column=close, shift_by=14)),
    ('hidden_drop_nans', drop_nan_rows()),
    ('hidden_feature_extraction', extract_columns(columns=['close', 'volume', 'sk5'])),
    ('hidden_to_matrix', to_float_matrix()),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```