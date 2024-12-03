### LINEAR REGRESSION
```yaml
model:
    type: regression
    name: sklearn_linear
    params:
        fit_intercept: True
```

### XG_BOOST REGRESSION
```yaml
model:
    type: regression
    name: xg_boost
    params:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.3
        subsample: 1.0
        colsample_bytree: 1.0
        gamma: 0.0
        reg_alpha: 0.0
        reg_lambda: 1.0
```

### SVR REGRESSION
```yaml
model:
    type: regression
    name: svr
    params:
        c_param: 1.0
        epsilon: 0.1
        kernel: rbf
        degree: 3
        gamma: scale
        coef0: 0
        shrinking: True
        tol: 1e-3
        max_iter: -1
```

### MLP REGRESSION
```yaml
model:
    type: regression
    name: mlp
    params:
        layer_sizes: 
            - 200
            - 100
            - 50
        activation_func: relu
        solver_func: adam
        alpha: 0.001
        learning_rate: adaptive
        max_iter: 1000
        early_stopping: True
        validation_fraction: 0.1
        tolerance: 1e-4
```

### CAT_BOOST REGRESSION
```yaml
model:
    type: regression
    name: cat_boost
    params:
        iterations: 100
        learning_rate: 0.1
        depth: 6
        l2_leaf_reg: 3.0
        loss_func: RMSE
```