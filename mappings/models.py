from common.misc import create_repository, option
from components.models import catboost_regressor, linear_regression, svr_regressor, mlp_regressor, xgboost_regressor

repository = create_repository({
    'linear_regression': option(
        linear_regression.custom_model,
        linear_regression.tests
    ),
    'xgboost_regressor': option(
        xgboost_regressor.custom_model,
        xgboost_regressor.tests
    ),
    'svr_regressor': option(
        svr_regressor.custom_model,
        svr_regressor.tests
    ),
    'catboost_regressor': option(
        catboost_regressor.custom_model,
        catboost_regressor.tests
    ),
    'mlp_regressor': option(
        mlp_regressor.custom_model,
        mlp_regressor.tests
    )
}, label='model')