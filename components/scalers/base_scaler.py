class base_scaler:
    def fit(self, features, labels=None):
        raise NotImplementedError()

    def transform(self, features):
        raise NotImplementedError()
    
    def __repr__(self):
        raise NotImplementedError()

    # INVOKED BY SKLEARN PIPELINES TO FIT & TRANSFORM SIMULTANOUSLY
    # THIS DOES NOT NEED TO BE MODIFIED
    def fit_transform(self, features, labels=None):
        self.fit(features)
        return self.transform(features)