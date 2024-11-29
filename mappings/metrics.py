from common.misc import create_repository, option
from components.metrics import regression, classification

repository = create_repository({

    # REGRESSION METRICS
    'regression.rse': option(regression.rse, None),
    'regression.mse': option(regression.mse, None),
    'regression.rmse': option(regression.rmse, None),
    'regression.mae': option(regression.mae, None),
    'regression.mape': option(regression.mape, None),
    'regression.smape': option(regression.smape, None),
    'regression.mase': option(regression.mase, None),

    # CLASSIFICATION METRICS
    'classification.accuracy': option(classification.accuracy, None),
    'classification.precision': option(classification.precision, None),
    'classification.recall': option(classification.recall, None),
    'classification.f_score': option(classification.f_score, None),
    'classification.roc_auc': option(classification.roc_auc, None),
    'classification.confusion_matrix': option(classification.confusion_matrix, None),

}, label='metric')