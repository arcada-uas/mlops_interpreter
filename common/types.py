from pydantic import BaseModel

########################################################################################
### EXPERIMENT YAML CONFIG SCHEMA

class name_params_pair(BaseModel):
    name: str
    params: dict

class method_params_pair(BaseModel):
    method: str
    params: dict

class dataset_schema(BaseModel):
    method: str
    params: dict
    expected_schema: dict[str, str]

class training_schema(BaseModel):
    feature_columns: list[str]
    label_column: str
    segmentation: method_params_pair
    scaler: name_params_pair
    model: name_params_pair

class config_schema(BaseModel):
    dataset: dataset_schema
    features: list[name_params_pair]
    training: training_schema