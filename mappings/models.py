from common.misc import create_repository, option
from components.models import cat_boost_regressor, linear_regression, xg_boost_regression, svr, mlp_regressor

repository = create_repository({
    'linear_regression': option(
        linear_regression.custom_model,
        linear_regression.tests
    ),
    'xg_boost_regression': option(
        xg_boost_regression.custom_model,
        xg_boost_regression.tests
    ),
    'svr': option(
        svr.custom_model,
        svr.tests
    ),
    'cat_boost_regressor': option(
        cat_boost_regressor.custom_model,
        cat_boost_regressor.tests
    ),
    'mlp_regressor': option(
        mlp_regressor.custom_model,
        mlp_regressor.tests
    )
}, label='model')