from common.misc import create_repository, option
from components.models import cat_boost_regressor, linear_regression, svr_regressor, xg_boost_regression, mlp_regressor

repository = create_repository({
    'linear_regression': option(
        linear_regression.custom_model,
        linear_regression.tests
    ),
    'xg_boost_regression': option(
        xg_boost_regression.custom_model,
        xg_boost_regression.tests
    ),
    'svr_regressor': option(
        svr_regressor.custom_model,
        svr_regressor.tests
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