from common.interpreter import create_repository, option
from components.models.regression import sklearn_linear, xg_boost, svr, cat_boost, mlp, svc
from components.models.classification import svc

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
    'regression.mlp': option(   
        mlp.custom_model,
        mlp.tests
    ),
    'regression.svr': option(
        svr.custom_model,
        svr.tests
    ),
    'regression.cat_boost': option(
        cat_boost.custom_model,
        cat_boost.tests
    ),
    'classification.svc': option(
        svc.custom_model,
        svc.tests
    ),
    # CLASSIFICATION MODELS
    # ...

}, label='model')