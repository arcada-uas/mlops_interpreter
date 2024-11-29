from pydantic import BaseModel, Field

########################################################################################
### EXPERIMENT YAML CONFIG SCHEMA

class model_schema(BaseModel):
    type: str = Field(min_length=3)
    name: str = Field(min_length=3)
    params: dict

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
    model: model_schema
    metrics: list[str] = Field(min_length=1)

class debug_schema(BaseModel):
    hide_traces: bool = False
    create_logfiles: bool = False
    limit_dataset: int = -1
    test_verbosity: int = Field(le=2, ge=0)

class config_schema(BaseModel):
    debug: debug_schema
    dataset: dataset_schema
    features: list[name_params_pair]
    training: training_schema