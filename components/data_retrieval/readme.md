### FROM CASSANDRA
```yaml
dataset:
    method: from_cassandra
    params:
        db_table: shohel.refined_stock_data
        stock_symbol: AAPL
        timestamps:
            start: '2019-01-01 00:00:00'
            end: '2019-01-10 00:00:00'
```