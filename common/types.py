from pydantic import BaseModel, Field

########################################################################################
### EXPERIMENT YAML CONFIG SCHEMA

class name_params_pair(BaseModel):
    name: str = Field(min_length=3)
    params: dict

class method_params_pair(BaseModel):
    method: str = Field(min_length=3)
    params: dict

class dataset_schema(BaseModel):
    method: str = Field(min_length=3)
    params: dict
    expected_schema: dict[str, str]

class training_schema(BaseModel):
    feature_columns: list[str] = Field(min_length=1)
    label_column: str = Field(min_length=3)
    segmentation: method_params_pair
    scaler: name_params_pair
    model: name_params_pair
    metrics: list[str] = Field(min_length=1)

class experiment_schema(BaseModel):
    hide_traces: bool

class config_schema(BaseModel):
    experiment: experiment_schema
    dataset: dataset_schema
    features: list[name_params_pair]
    training: training_schema