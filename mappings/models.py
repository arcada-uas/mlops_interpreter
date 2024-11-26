from common.misc import create_repository, option
from components.models import linear_regression, xg_boost_regression

repository = create_repository({
    'linear_regression': option(
        linear_regression.custom_model,
        linear_regression.tests
    ),
    'xg_boost_regression': option(
        xg_boost_regression.custom_model,
        xg_boost_regression.tests
    )
}, label='model')