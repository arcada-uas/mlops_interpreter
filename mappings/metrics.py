from common.misc import create_repository, option
from sklearn.metrics import r2_score, mean_squared_error

# MSE NUMBERS ARE VERY SMALL
# MULTIPLY BY 10^7 TO INCREASE VISIBILITY
enhanced_mse = lambda labels, predictions: mean_squared_error(labels, predictions) * 10**7

repository = create_repository({

    # REGRESSION METRICS
    'root_squared': option(r2_score, None),
    'mean_squared': option(enhanced_mse, None),

    # CLASSIFICATION METRICS
    # ...

}, label='metric')