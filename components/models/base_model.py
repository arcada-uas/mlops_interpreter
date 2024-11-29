from mappings.metrics import repository as metric_options

class base_model:
    def __init__(self, input_params: dict):
        self.model = None
        self.metrics = []

    def __repr__(self):
        raise NotImplementedError()

    def fit(self, features, labels=None):
        raise NotImplementedError()

    def predict(self, features):
        raise NotImplementedError()

    # INVOKED BY INTERPRETER DURING RUN SCRIPT
    def set_metrics(self, metrics: list[str]):
        self.metrics = metrics

    # UNIFIED SCORE FUNCTION FOR ALL MODELS
    # DO NOT OVERWRITE THIS    
    def score(self, features, labels):
        assert self.model != None, 'A MODEL HAS NOT BEEN TRAINED YET'
        assert len(self.metrics) > 0, 'NO MODEL METRICS SPECIFIED'

        # GENERATE PREDICTIONS WITH MODEL
        predictions = self.predict(features)
        container = {}

        for metric_name in self.metrics:
            metric_func = metric_options.get(metric_name)

            # ROUND VALUE TO 4TH DECIMAL FOR READABILITY
            rounded_value = round(metric_func(labels, predictions), 4)
            container[metric_name] = rounded_value

        return container
