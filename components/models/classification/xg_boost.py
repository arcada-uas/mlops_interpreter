from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from xgboost import XGBClassifier


class xgboost_classification_schema(base_schema):
    n_estimators: int = Field(ge=1, description="Number of trees in the ensemble.")
    max_depth: int = Field(ge=1, description="Maximum depth of the trees.")
    learning_rate: float = Field(gt=0, description="Boosting learning rate.")
    subsample: float = Field(gt=0, le=1, description="Subsample ratio of the training instances.")
    eval_metric: str = Field(description="Evaluation metric for training.")
    random_state: int = Field(description="Seed for reproducibility.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, n_estimators: int, max_depth: int, learning_rate: float, subsample: float, eval_metric: str, random_state: int):
        # VALIDATE INPUTS
        params = xgboost_classification_schema(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            eval_metric=eval_metric,
            random_state=random_state,
        )

        # SAVE MODEL PARAMS IN STATE
        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample
        self.eval_metric = params.eval_metric
        self.random_state = params.random_state

        # SHOULD ALWAYS DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f"xgboost_classification({self.stringify_vars(['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'eval_metric', 'random_state'])})"

    def fit(self, features: list[list[float]], labels: list[int] = None):
        self.pre_fitting_asserts(features, labels)

        # CREATE THE MODEL
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
        )

        # TRAIN IT
        self.model.fit(features, labels)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):
        """
        Test model instantiation with static parameters.
        """
        # CREATE THE MODEL
        model = custom_model(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric="logloss",
            random_state=42,
        )

        # VERIFY PARAMETERS
        self.assertEqual(model.n_estimators, 100, "Mismatch in n_estimators.")
        self.assertEqual(model.max_depth, 6, "Mismatch in max_depth.")
        self.assertEqual(model.learning_rate, 0.1, "Mismatch in learning_rate.")
        self.assertEqual(model.subsample, 0.8, "Mismatch in subsample.")
        self.assertEqual(model.eval_metric, "logloss", "Mismatch in eval_metric.")
        self.assertEqual(model.random_state, 42, "Mismatch in random_state.")

    def test_01_runs_with_yaml_params(self):
        """
        Test model instantiation with dynamically loaded YAML parameters.
        """
        custom_model(**self.yaml_params)
