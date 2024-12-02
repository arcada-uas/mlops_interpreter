from common.pydantic import base_schema, Field
from components.models.base_model import base_model
from common.testing import base_unittest
from sklearn.svm import SVC

class svc_classification_schema(base_schema):
    c_param: float = Field(gt=0, lt=1, description="Regularization parameter for SVC.")
    kernel: str = Field(description="Kernel type ('linear', 'poly', 'rbf', 'sigmoid').")
    probability: bool = Field(description="Enable probability estimates.")

##############################################################################################################
##############################################################################################################

class custom_model(base_model):
    def __init__(self, c_param: float, kernel: str, probability: bool):

        # VALIDATE INPUTS
        params = svc_classification_schema(c_param, kernel, probability)

        # SAVE MODEL PARAMS IN STATE
        self.c_param = params.c_param
        self.kernel = params.kernel
        self.probability = params.probability

        # SHOULD ALWAYS DEFAULT TO NONE
        self.model = None

    def __repr__(self):
        params_str = self.stringify_vars(['c_param', 'kernel', 'probability'])
        return f"svc_classification({ params_str })"

    def fit(self, features: list[list[float]], labels: list[int] = None):
        self.pre_fitting_asserts(features, labels)

        # CREATE THE MODEL
        self.model = SVC(
            C=self.c_param,
            kernel=self.kernel,
            probability=self.probability,
            random_state=42
        )

        # TRAIN IT
        self.model.fit(features, labels)

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_runs_with_static_params(self):
        pass