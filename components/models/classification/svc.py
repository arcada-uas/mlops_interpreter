from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.svm import SVC


class svc_classification_schema(base_schema):
    C: float = Field(gt=0, description="Regularization parameter for SVC.")
    kernel: str = Field(description="Kernel type ('linear', 'poly', 'rbf', 'sigmoid').")
    probability: bool = Field(description="Enable probability estimates.")
    random_state: int = Field(description="Seed for reproducibility.")


##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, C: float, kernel: str, probability: bool, random_state: int):
        # VALIDATE INPUTS
        params = svc_classification_schema(C=C, kernel=kernel, probability=probability, random_state=random_state)

        # SAVE MODEL PARAMS IN STATE
        self.C = params.C
        self.kernel = params.kernel
        self.probability = params.probability
        self.random_state = params.random_state

        # SHOULD ALWAYS DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        return f"svc_classification({self.stringify_vars(['C', 'kernel', 'probability', 'random_state'])})"

    def fit(self, features: list[list[float]], labels: list[int] = None):
        self.pre_fitting_asserts(features, labels)

        # CREATE THE MODEL
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
            random_state=self.random_state
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
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42
        )

        # VERIFY PARAMETERS
        self.assertEqual(model.C, 1.0, "Mismatch in C.")
        self.assertEqual(model.kernel, "rbf", "Mismatch in kernel.")
        self.assertTrue(model.probability, "Mismatch in probability.")
        self.assertEqual(model.random_state, 42, "Mismatch in random_state.")

    def test_01_runs_with_yaml_params(self):
        """
        Test model instantiation with dynamically loaded YAML parameters.
        """
        custom_model(**self.yaml_params)
