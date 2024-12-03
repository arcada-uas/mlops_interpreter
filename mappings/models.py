from common.interpreter import create_repository, option
from components.models.regression import sklearn_linear, xg_boost as xg_reg, svr, cat_boost, mlp
from components.models.classification import svc, xg_boost

repository = create_repository({

    ##############################################################
    ### REGRESSION MODELS
    
    'regression.sklearn_linear': option(
        sklearn_linear.custom_model,
        sklearn_linear.tests
    ),
    'regression.xg_boost': option(
        xg_reg.custom_model,
        xg_reg.tests
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
    
    # CLASSIFICATION MODELS
    'classification.svc': option(
        svc.custom_model,
        svc.tests
    ),
    'classification.xg_boost': option(
        xg_boost.custom_model,
        xg_boost.tests
    ),

}, label='model')