from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from catboost import CatBoostRegressor

# model:
#     type: regression
#     name: cat_boost
#     params:
#         iterations: 100
#         learning_rate: 0.1
#         depth: 6
#         l2_leaf_reg: 3.0
#         loss_func: RMSE

class cat_boost_schema(base_schema):
    iterations: int = Field(ge=1, description="The number of trees to be built.")
    learning_rate: float = Field(gt=0, description="The learning rate for the model.")
    depth: int = Field(ge=1, description="The depth of the trees.")
    l2_leaf_reg: float = Field(gt=0, description="L2 regularization coefficient.")
    loss_func: str = Field(description="The loss function to be used.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, iterations: int, learning_rate: float, depth: int, l2_leaf_reg: float, loss_func: str):

        # VALIDATE PARAMS
        params = cat_boost_schema(iterations, learning_rate, depth, l2_leaf_reg, loss_func)

        # STORE MODEL PARAMS IN STATE
        self.iterations = params.iterations
        self.learning_rate = params.learning_rate
        self.depth = params.depth
        self.l2_leaf_reg = params.l2_leaf_reg
        self.loss_func = params.loss_func

        # SHOULD DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f'cat_boost_regression({
            self.stringify_vars([
                'iterations', 'learning_rate', 'depth',
                'l2_leaf_reg', 'loss_func',
            ])
        })'

    def fit(self, features: list[list[float]], labels: list[float] = None):
        self.pre_fitting_asserts(features, labels)
        
        # CREATE THE MODEL
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_func,
            random_seed=42,
            verbose=2,
            allow_writing_files=False
        )

        # TRAIN IT
        self.model.fit(features, labels)
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):

        # CREATE THE MODELS
        model = custom_model(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            loss_func='RMSE',
        )

        # MAKE SURE PARAMS WERE SET CORRECTLY
        self.assertEqual(model.iterations, 100, "Mismatch in iterations")
        self.assertEqual(model.learning_rate, 0.1, "Mismatch in learning_rate")
        self.assertEqual(model.depth, 6, "Mismatch in depth")
        self.assertEqual(model.l2_leaf_reg, 3.0, "Mismatch in l2_leaf_reg")
        self.assertEqual(model.loss_func, "RMSE", "Mismatch in loss_function")

    def test_01_runs_with_yaml_params(self):
        custom_model(**self.yaml_params)