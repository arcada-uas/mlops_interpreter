class base_feature:

    # REQUIRED METHOD, EVEN THO IT DOES NOTHING FOR FEATURES
    # ...TO COMPLY WITH SKLEARN PIPELINES
    def fit(self, features, labels=None):
        return self
    
    def __repr__(self):
        raise NotImplementedError()
    
    def transform(self, features):
        raise NotImplementedError()