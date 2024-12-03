## OVERVIEW
todo

## IMPLEMENTED

- [Data retrieval methods](components/data_retrieval/)
    - Cassandra DB.
    - CSV files.
- [Segmentation methods](components/segmentation/):
    - Standard train/test/validation split.
- [Features](components/features/):
    - Stochastic K.
    - Column shifting.
- [Scalers](components/scalers/):
    - Standard scaler.
    - MinMax scaler.
- [Models](components/models/):
    - Regression:
        - XG Boost
        - Sklearn linear
        - MLP
        - SVR
        - Cat Boost
- [Metrics](components/metics/):
    - Regression:
        - RSE.
        - MSE.
        - RMSE.
        - MAE.
        - MAPE.
        - SMAPE.
        - MASE.
    - Classification:
        - Accuracy.
        - Precision.
        - Recall.
        - F-score.
        - ROC_AUC.
        - Confusion Matrix.

## WORKFLOW
1. Implement your functionality in `components/`.
2. Make it available through to the interpreter in `mappings/`:
    - [Data retrieval methods](mappings/data_retrieval.py)
    - [Segmentation methods](mappings/segmentation.py)
    - [Features](mappings/features.py)
    - [Scalers](mappings/scalers.py)
    - [Models](mappings/models.py)
    - [Metrics](mappings/metrics.py)
3. Reference your functionality in the YAML configuration file.
    - Default: [`pipeline.yaml`](pipeline.yaml)
    - More examples in `runs/`
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
### YAML DECLARATION
```yaml
# WHERE DO WE FETCH THE DATASET FROM?
dataset:
    method: from_cassandra
    params:
        db_table: shohel.refined_stock_data
        stock_symbol: AAPL
        timestamps:
            start: '2019-01-01 00:00:00'
            end: '2019-01-10 00:00:00'

    # WHAT COLUMNS ARE EXPECTED?
    expected_schema:
        symbol: str
        timestamp: str
        open: float
        close: float
        high: float
        low: float
        volume: int
```
```yaml
# WHAT FEATURES SHOULD WE APPLY?
features:
    -   method: stochastic_k
        params:
            window_size: 5
            output_column: sk5
    -   method: shift_column
        params:
            target_column: close
            shift_by: 14
            output_column: shifted_close
```
```yaml
training:

    # WHAT COLUMNS SHOULD WE TRAIN ON?
    feature_columns:
        - open
        - close
        - high
        - low
        - volume
        - sk5

    # WHAT COLUMN CONTAINS THE LABEL?
    label_column: shifted_close

    # HOW SHOULD WE SEGMENT THE DATASET?
    segmentation:
        method: standard_ttv
        params:
            segments:
                - train: 0.75
                - test: 0.15
                - validate: 0.1

    # HOW SHOULD WE SCALE THE FEATURES?
    scaler:
        method: standard_scaler
        params:
            use_std: True
            use_mean: True

    # WHAT MODEL SHOULD WE USE?
    model:
        type: regression
        method: cat_boost
        params:
            iterations: 50
            learning_rate: 0.1
            depth: 6
            l2_leaf_reg: 3.0
            loss_func: RMSE
    
    # WHAT METRICS SHOULD THE MODEL BE EVALUATED WITH?
    metrics:
        - rse
        - mse
        - rmse
```

### UNITTEST PIPELINE
```sh
make test

# EXAMPLE OUTPUT
25/25 UNITTESTS PASSED -- SHOWING COVERAGE
--------------------------------------------------------------------
Name                                             Stmts   Miss  Cover
--------------------------------------------------------------------
components/data_retrieval/from_cassandra.py         50     22    56%
components/features/hidden/drop_nan_rows.py         19     11    42%
components/features/hidden/extract_features.py      28     15    46%
components/features/hidden/to_dataframe.py          22     10    55%
components/features/standard/shift_column.py        41     18    56%
components/features/standard/stochastic_k.py        55     17    69%
components/models/regression/cat_boost.py           47     25    47%
components/scalers/minmax_scaler.py                 35     17    51%
components/segmentation/standard_ttv.py             64     15    77%
--------------------------------------------------------------------
TOTAL                                              361    150    58%
```

### ASSEMBLE PIPELINE
```sh
make assemble

# EXAMPLE OUTPUT
ASSEMBLING PIPELINE
--------------------------------------------------------------------
[1/9] FETCHED DATASET:                                         n=250
[2/9] CREATED 'train' SEGMENT:                                 n=188
[2/9] CREATED 'test' SEGMENT:                                   n=38
[2/9] CREATED 'validate' SEGMENT:                               n=24
[3/9] ADDED HIDDEN FEATURE:                             to_dataframe
[4/9] ADDED STANDARD FEATURE:                           stochastic_k
[4/9] ADDED STANDARD FEATURE:                           shift_column
[5/9] ADDED HIDDEN FEATURE:                            drop_nan_rows
[6/9] EXTRACTED LABELS:                                shifted_close
[7/9] ADDED HIDDEN FEATURE:                         extract_features
[8/9] ADDED SCALER:                                    minmax_scaler
[9/9] ADDED MODEL:                              regression.cat_boost
```

### TEST -> ASSEMBLE -> TRAIN -> ANALYZE PIPELINE
```sh
make complete

# EXAMPLE OUTPUT
TRAINING PIPELINE
--------------------------------------------------------------------
0:	learn: 0.0010099	total: 47ms	    remaining: 2.3s
2:	learn: 0.0009102	total: 47.7ms	remaining: 748ms
4:	learn: 0.0008250	total: 48.4ms	remaining: 435ms
6:	learn: 0.0007528	total: 48.8ms	remaining: 300ms
8:	learn: 0.0006993	total: 49.2ms	remaining: 224ms
10:	learn: 0.0006573	total: 49.7ms	remaining: 176ms
12:	learn: 0.0006119	total: 50.1ms	remaining: 143ms
14:	learn: 0.0005829	total: 50.5ms	remaining: 118ms
16:	learn: 0.0005557	total: 50.9ms	remaining: 98.8ms
18:	learn: 0.0005370	total: 51.3ms	remaining: 83.6ms
20:	learn: 0.0005202	total: 51.6ms	remaining: 71.3ms
22:	learn: 0.0005054	total: 52ms	    remaining: 61.1ms
24:	learn: 0.0004929	total: 52.4ms	remaining: 52.4ms
26:	learn: 0.0004811	total: 52.9ms	remaining: 45ms
28:	learn: 0.0004712	total: 53.3ms	remaining: 38.6ms
30:	learn: 0.0004638	total: 53.6ms	remaining: 32.9ms
32:	learn: 0.0004526	total: 54ms	    remaining: 27.8ms
34:	learn: 0.0004444	total: 54.4ms	remaining: 23.3ms
36:	learn: 0.0004404	total: 54.8ms	remaining: 19.2ms
38:	learn: 0.0004324	total: 55.2ms	remaining: 15.6ms
40:	learn: 0.0004263	total: 55.5ms	remaining: 12.2ms
42:	learn: 0.0004222	total: 56ms	    remaining: 9.11ms
44:	learn: 0.0004161	total: 56.4ms	remaining: 6.26ms
46:	learn: 0.0004104	total: 56.8ms	remaining: 3.62ms
48:	learn: 0.0004025	total: 57.2ms	remaining: 1.17ms
49:	learn: 0.0003997	total: 57.5ms	remaining: 0us

METRIC: regression.rse
--------------------------------------------------------------------
TRAIN:                                                        0.8601
TEST:                                                       -45.1741
VALIDATE:                                                  -112.1963

METRIC: regression.mse
--------------------------------------------------------------------
TRAIN:                                                        1.5979
TEST:                                                        51.1833
VALIDATE:                                                    44.0019

METRIC: regression.rmse
--------------------------------------------------------------------
TRAIN:                                                        1.2641
TEST:                                                         7.1542
VALIDATE:                                                     6.6334
```