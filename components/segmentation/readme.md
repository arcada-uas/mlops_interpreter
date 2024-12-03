### STANDARD TRAIN/TEST/VALIDATION SPLIT
```yaml
training:
    segmentation:
        method: standard_ttv
        params:
            segments:
                - train: 0.75
                - test: 0.15
                - validate: 0.1
```