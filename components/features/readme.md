### STOCHASTIC K
```yaml
features:
    -   name: stochastic_k
        params:
            window_size: 5
            output_column: sk5
```

### SHIFTING COLUMN
```yaml
features:
    -   name: shift_column
        params:
            target_column: close
            shift_by: 14
            output_column: shifted_close
```