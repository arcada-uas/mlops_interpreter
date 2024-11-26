class base_model:
    def __init__(self, input_params: dict):

        # NON NEGATIVE FLOAT
        self.alpha = input_params['alpha']

    def fit(self, features, labels=None):
        raise NotImplementedError()

    def predict(self, features):
        raise NotImplementedError()
    
    def score(self, features, labels):
        raise NotImplementedError()