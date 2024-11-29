from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math, numpy as np

##############################################################################################################
##############################################################################################################

rse = lambda actual, predictions: r2_score(actual, predictions)
mse = lambda actual, predictions: float(mean_squared_error(actual, predictions) * 10**7)
mae = lambda actual, predictions: float(mean_absolute_error(actual, predictions) * 10**4)
rmse = lambda actual, predictions: math.sqrt(mse(actual, predictions))

##############################################################################################################
##############################################################################################################

def mape(actual, predictions):
    metric_value = np.mean(np.abs((actual - predictions) / actual)) * 100
    return float(metric_value * 10**2)

##############################################################################################################
##############################################################################################################

def smape(actual, predictions):
    metric_value = np.mean(2 * np.abs(actual - predictions) / (np.abs(actual) + np.abs(predictions))) * 100
    return float(metric_value * 10**2)

##############################################################################################################
##############################################################################################################

def mase(actual, predictions):
    naive_forecast_error = np.mean(np.abs(np.diff(actual))) # Naive forecast using lag-1
    metric_value = np.mean(np.abs(actual - predictions)) / naive_forecast_error
    return float(metric_value)