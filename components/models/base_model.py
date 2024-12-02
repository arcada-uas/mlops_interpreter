from abc import ABC, abstractmethod
from mappings.metrics import repository as metric_options

class base_model:
    def __init__(self):
        self.model = None
        self._metrics = []
        self._prediction_window = 0

    def __repr__(self):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, features, labels=None):
        raise NotImplementedError()

    # IF YOUR PREDICT METHOD DOES NOT DO ANYTHING SPECIAL
    # THIS DOES NOT NEED TO CHANGED
    def predict(self, features):
        assert self.model is not None, 'A MODEL HAS NOT BEEN TRAINED YET'
        return self.model.predict(features)

    ##############################################################################################################
    ### SHARED BOILERPLATE FUNCTIONALITY FOR ALL MODELS

    # INVOKED BY INTERPRETER DURING RUN SCRIPT
    # SAVE LIST OF METRICS FROM YAML IN MODEL STATE
    def set_metrics(self, metrics: list[str]):
        self._metrics = metrics

    # UNIFIED SCORE FUNCTION FOR ALL MODELS
    # DO NOT OVERWRITE THIS
    @abstractmethod
    def score(self, features, labels):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        assert len(self._metrics) > 0, 'NO MODEL METRICS SPECIFIED'

        # GENERATE PREDICTIONS WITH MODEL
        predictions = self.predict(features)
        container = {}

        for metric_name in self._metrics:
            metric_func = metric_options.get(metric_name)

            # ROUND VALUE TO 4TH DECIMAL FOR READABILITY
            rounded_value = round(metric_func(labels, predictions), 4)
            container[metric_name] = rounded_value

        return container

    # INVOKED BY INTERPRETER DURING RUN SCRIPT
    # SAVE MINIMUM ROW COUNT TO CREATE ONE FULL ROW OF FEATURES
    # REQUIRED TO MAKE ONE PREDICTION
    def set_prediction_window(self, window_size: int):
        self._prediction_window = window_size

    # STRINGIFY A LIST OF CLASS VARS
    # SHORTHAND FOR __REPR__ METHOD
    def stringify_vars(self, var_names):
        for name in var_names:
            assert name in self.__dict__, f"(__REPR__ ERROR) MODEL VARIABLE '{name}' DOES NOT EXIST"
        return ', '.join([f'{x}={self.__dict__[x]}' for x in var_names])
    
    # ALMOST EVERY FIT IMPLEMENTATION SHARES THESE PRE-REQ ASSERTS
    def pre_fitting_asserts(self, features, labels):
        assert self.model is None, "A MODEL HAS ALREADY BEEN TRAINED."
        assert features is not None and labels is not None, "Features and labels cannot be None."
        assert len(features) > 0, "Features cannot be empty."
        assert len(features) == len(labels), "Features and labels must have the same length."