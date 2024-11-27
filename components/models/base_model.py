from abc import ABC, abstractmethod
from pydantic import BaseModel as PydanticModel, Field


class BaseModel(ABC):
    class Config(PydanticModel):
        alpha: float = Field(
            ..., ge=0, description="Alpha must be a non-negative float"
        )

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def __init__(self, input_params: dict):
        config = self.Config(**input_params)
        self.alpha = config.alpha

    @abstractmethod
    def fit(self, features, labels=None):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def score(self, features, labels):
        pass
