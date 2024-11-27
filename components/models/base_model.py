from abc import ABC, abstractmethod


class base_model(ABC):
    @abstractmethod
    def fit(self, features, labels=None):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def score(self, features, labels):
        pass
