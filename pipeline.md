```yaml
dataset:

    # WHERE DO WE FETCH THE DATASET FROM?
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
features:

    # ADD FIRST FEATURE
    -   method: stochastic_k
        params:
            window_size: 5
            output_column: sk5

    # ADD SECOND FEATURE
    -   method: shift_column
        params:
            target_column: close
            shift_by: 14
            output_column: shifted_close

    # ADD SECOND FEATURE
    -   method: to_categorical
        params:
            columns:
                first: shifted_close
                second: close
            categories:
                gt: buy
                lt: sell
                fallback: hold
            output_column: categorical_close
```
```yaml
pre_processing:

    # WHAT COLUMNS CONTAIN THE TRAINING FEATURES?
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
    scaling:
        method: standard_scaler
        params:
            use_std: True
            use_mean: True
```
```yaml
training:

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

    # WHERE SHOULD WE TRAIN THE MODEL?
    # USING RAY CLUSTER
    hardware_device: gpu_4090
    # hardware_device: cpu_i7

    # HOW SHOULD THE PIPELINE BE STORED?
    # USING MLFLOW
    mlflow:
        experiment_name: bla
```