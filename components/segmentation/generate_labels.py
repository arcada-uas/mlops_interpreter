from common.testing import base_unittest, validate_params
from pydantic import BaseModel, Field

class input_schema(BaseModel):
    datasets: dict[str, list[dict]]
    pipeline: list[tuple]
    label_column: str = Field(min_length=3)
    feature_columns: list[str] = Field(min_length=1)

##################################################################################################
##############################################################################################################

# APPLY PIPELINE FEATURES ON SUBSET
# AND EXTRACT LABEL COLUMN FROM THE RESULTING DATAFRAME
def generate_segment_labels(input_params: dict):
    params = validate_params(input_params, input_schema)
    container = {}

    for segment_name, subset in params.datasets.items():

        # TODO: TEST IF THIS IS NECESSARY
        # TODO: TEST IF THIS IS NECESSARY
        # TODO: TEST IF THIS IS NECESSARY
        cloned_dataset = [x for x in subset]

        # APPLY EACH FEATURE
        for _, feature in params.pipeline:
            cloned_dataset = feature.transform(cloned_dataset)

        # MAKE SURE LABEL COLUMN EXISTS
        df_columns = list(cloned_dataset.columns)
        assert params.label_column in df_columns, f"LABEL COLUMN '{params.label_column}' DOES NOT EXIST\nOPTIONS: {df_columns}"
        
        # EXTRACT THE LABEL COLUMN
        labels = cloned_dataset[params.label_column].tolist()
        container[segment_name] = labels

        # WORKAROUND: TO FIX ISSUE WITH CREATE_PIPELINE SCRIPT WHEN USERS HAS MALFORMED FEATURE COLUMNS
        # MAKE SURE ALL FEATURE COLUMNS EXIST
        for column_name in params.feature_columns:
            assert column_name in df_columns, f"FEATURE COLUMN '{column_name}' DOES NOT EXIST\nOPTIONS: {df_columns}"

    return container

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_schema(self):
        pass