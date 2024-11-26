from pydantic import BaseModel, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

class input_schema(BaseModel):
    n_estimators: int = Field(..., ge=1, description="Number of trees in the XGBoost model")
    max_depth: int = Field(..., ge=1, description="Maximum depth of the trees")
    learning_rate: float = Field(..., gt=0, description="Learning rate for the model")
    subsample: float = Field(..., gt=0, le=1, description="Subsample ratio of the training data")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, input_params: dict):

        # VALIDATE INPUT PARAMS
        assert isinstance(input_params, dict), f"ARG 'input_params' MUST BE OF TYPE DICT, GOT {type(input_params)}"
        params = input_schema(**input_params)

        # SAVE PARAMS IN STATE
        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample

        # MAKE SURE MODEL DEFAULTS TO NONE
        self.model = None

    def __repr__(self):
        return f"""
        xgboost(
            n_estimators={self.n_estimators}
            max_depth={self.max_depth}
            learning_rate={self.learning_rate}
            subsample={self.subsample}
        )    
        """

    def fit(self, features, labels=None):
        assert self.model == None, 'A MODEL HAS ALREADY BEEN TRAINED'

        # FIT THE MODEL
        xgboost_model = XGBRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            learning_rate=self.learning_rate, 
            subsample=self.subsample,
            n_jobs=-1
        ).fit(features, labels)

        # SAVE IT IN STATE
        self.model = xgboost_model

    def predict(self, features):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)
    
    # def score(self, features, labels):
    #     assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'        
        
    #     # MAKE PREDICTIONS ON GIVEN FEATURES        
    #     predictions = self.model.predict(features)        
        
    #     # COMPUTE SOME METRICS        
    #     return {
    #         'r2': round(r2_score(labels, predictions), 4),
    #         'mse': round(float(mean_squared_error(labels, predictions)), 4)
    #     }
        
    def score(self, features, labels):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'

        # MAKE PREDICTIONS ON GIVEN FEATURES
        predictions = self.model.predict(features)

        # COMPUTE SOME METRICS
        return {
            'r2': r2_score(labels, predictions),
            'mse': mean_squared_error(labels, predictions)
        }
    
##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_validate_input(self):
        custom_model(self.input_params)