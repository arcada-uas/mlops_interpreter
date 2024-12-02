from common.testing import base_unittest
from common.pydantic import base_schema, Field

class generate_labels_schema(base_schema):
    datasets: dict[str, list[dict]] = Field(min_length=1)
    pipeline: list[tuple] = Field(min_length=1)
    label_column: str = Field(min_length=3)
    feature_columns: list[str] = Field(min_length=1)

##############################################################################################################
##############################################################################################################

# APPLY PIPELINE FEATURES ON SUBSET
# AND EXTRACT LABEL COLUMN FROM THE RESULTING DATAFRAME
def generate_segment_labels(datasets: dict, pipeline: list, label_column: str, feature_columns: list):
    params = generate_labels_schema(datasets, pipeline, label_column, feature_columns)
    container = {}
    feature_buffers = {}

    for segment_name, subset in params.datasets.items():

        # TODO: TEST IF THIS IS NECESSARY
        # TODO: TEST IF THIS IS NECESSARY
        # TODO: TEST IF THIS IS NECESSARY
        cloned_dataset = [x for x in subset]
        
        original_length = len(cloned_dataset)

        # APPLY EACH FEATURE
        for _, feature in params.pipeline:
            cloned_dataset = feature.transform(cloned_dataset)

        length_error = f"SEGMENT '{segment_name}' IS TOO SMALL TO PRODUCE ONE FULL ROW OF FEATURES.\nSOLUTION: INCREASE PERCENTAGE OR SET TO ZERO."
        assert len(cloned_dataset) > 0, length_error

        # FIND HOW MANY ROWS WERE CUT FORM THE DATASET
        # DUE TO FEATURE WINDOWS
        feature_buffers[segment_name] = original_length - len(cloned_dataset)

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

    # FIND HOW MANY ROWS ARE REQUIRED TO MAKE ONE FULL ROW OF FEATURES
    minimum_row_count = max(feature_buffers.values())
    
    return container, minimum_row_count

##############################################################################################################
##############################################################################################################

class tests(base_unittest):
    def test_00_input_schema(self):
        pass