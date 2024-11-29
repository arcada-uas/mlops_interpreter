from common.misc import create_repository, option
from components.models.regression import sklearn_linear, xg_boost

repository = create_repository({

    # REGRESSION MODELS
    'regression.sklearn_linear': option(
        sklearn_linear.custom_model,
        sklearn_linear.tests
    ),
    'regression.xg_boost': option(
        xg_boost.custom_model,
        xg_boost.tests
    ),

    # CLASSIFICATION MODELS
    # ...

}, label='model')