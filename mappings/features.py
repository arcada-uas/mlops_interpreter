from common.interpreter import create_repository, option
from components.features.standard import shift_column, stochastic_k, buy_sell, log_return, quartile, oracle
from components.features.hidden import drop_nan_rows, extract_features, to_dataframe

repository = create_repository({

    # STANDARD FEATURES
    'shift_column': option(
        shift_column.custom_feature,
        shift_column.tests
    ),
    'stochastic_k': option(
        stochastic_k.custom_feature,
        stochastic_k.tests
    ),
    'buy_sell': option(
        buy_sell.custom_feature,
        buy_sell.tests
    ),
    'log_return': option(
        log_return.custom_feature,
        log_return.tests
    ),
    'quartile': option(
        quartile.custom_feature,
        quartile.tests
    ),
    'oracle': option(
        oracle.custom_feature,
        oracle.tests
    ),
    
    # "HIDDEN" FEATURES
    'to_dataframe': option(
        to_dataframe.custom_feature,
        to_dataframe.tests
    ),
    'drop_nan_rows': option(
        drop_nan_rows.custom_feature,
        drop_nan_rows.tests
    ),
    'extract_features': option(
        extract_features.custom_feature,
        extract_features.tests
    ),
}, label='feature')