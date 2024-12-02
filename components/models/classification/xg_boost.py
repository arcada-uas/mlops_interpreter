from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


class xgboost_classification_schema(base_schema):
    n_estimators: int = Field(ge=1, description="Number of trees in the ensemble.")
    max_depth: int = Field(ge=1, description="Maximum depth of the trees.")
    learning_rate: float = Field(gt=0, description="Boosting learning rate.")
    subsample: float = Field(gt=0, le=1, description="Subsample ratio of the training instances.")
    eval_metric: str = Field(description="Evaluation metric for training.")
    random_state: int = Field(description="Seed for reproducibility.")
    use_label_encoder: bool = Field(description="Whether to use the label encoder in XGBoost.")


##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, n_estimators: int, max_depth: int, learning_rate: float, subsample: float, eval_metric: str, random_state: int, use_label_encoder: bool):
        # VALIDATE INPUTS
        params = xgboost_classification_schema(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            eval_metric=eval_metric,
            random_state=random_state,
            use_label_encoder=use_label_encoder
        )

        # SAVE MODEL PARAMS IN STATE
        self.n_estimators = params.n_estimators
        self.max_depth = params.max_depth
        self.learning_rate = params.learning_rate
        self.subsample = params.subsample
        self.eval_metric = params.eval_metric
        self.random_state = params.random_state
        self.use_label_encoder = params.use_label_encoder

        # SHOULD ALWAYS DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f"xgboost_classification({self.stringify_vars(['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'eval_metric', 'random_state', 'use_label_encoder'])})"

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
            use_label_encoder=self.use_label_encoder
        )

        # TRAIN IT
        self.model.fit(features, labels)

    def predict(self, features: list[list[float]]) -> list[int]:
        self.pre_prediction_asserts(features)
        return self.model.predict(features)

    def calculate_metrics(self, labels: list[int], predictions: list[int]) -> dict:
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average='weighted'),
            "recall": recall_score(labels, predictions, average='weighted'),
            "f1_score": f1_score(labels, predictions, average='weighted'),
            "roc_auc": roc_auc_score(labels, predictions) if len(set(labels)) == 2 else None,
            "specificity": self.calculate_specificity(labels, predictions)
        }
        return metrics

    def calculate_specificity(self, labels: list[int], predictions: list[int]) -> float:
        cm = confusion_matrix(labels, predictions)
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else None


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
            use_label_encoder=False
        )

        # VERIFY PARAMETERS
        self.assertEqual(model.n_estimators, 100, "Mismatch in n_estimators.")
        self.assertEqual(model.max_depth, 6, "Mismatch in max_depth.")
        self.assertEqual(model.learning_rate, 0.1, "Mismatch in learning_rate.")
        self.assertEqual(model.subsample, 0.8, "Mismatch in subsample.")
        self.assertEqual(model.eval_metric, "logloss", "Mismatch in eval_metric.")
        self.assertEqual(model.random_state, 42, "Mismatch in random_state.")
        self.assertFalse(model.use_label_encoder, "Mismatch in use_label_encoder.")

    def test_01_runs_with_yaml_params(self):
        """
        Test model instantiation with dynamically loaded YAML parameters.
        """
        custom_model(**self.yaml_params)
