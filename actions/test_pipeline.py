from common.testing import run_tests, build_and_validate_schema
from common import files, types
from pandas import DataFrame

from mappings.data_retrieval import repository as retrieval_options
from mappings.features import repository as feature_options
from mappings.models import repository as model_options
from mappings.scalers import repository as scaler_options
from mappings.segmentation import repository as segmentation_options

##############################################################################################################
##############################################################################################################

def run(yaml_config: types.config_schema):
    try:

        # KEEP TRACK OF HOW MANY TESTS HAVE BEEN RAN
        num_tests_ran: int = 0

        # HIDE ERROR TRACES TO MAKE ASSERT ERRORS
        # EASIER TO COMPREHEND AND LOCATE
        if yaml_config.debug.hide_traces:
            import sys
            sys.tracebacklimit = 0

    #####################################################################################
    ### UNITTEST DATASET RETRIEVAL

        dataset_module = retrieval_options.get_tests(yaml_config.dataset.method)
        num_tests_ran += run_tests(dataset_module, yaml_config.debug.test_verbosity, yaml_config.dataset.params)

        # THE TESTS PASSED -- LOAD IN A SAMPLE OF THE REAL DATASET FOR OTHER TESTS
        fetching_func = retrieval_options.get(yaml_config.dataset.method)
        sample_dataset = DataFrame(fetching_func(yaml_config.dataset.params, unittest_limit=200))

        # SELECT A SAMPLE ROW & MAKE SURE IT FOLLOWS EXPECTED SCHEMA
        sample_row = sample_dataset.iloc[0].to_dict()
        build_and_validate_schema(sample_row, yaml_config.dataset.expected_schema)

    #####################################################################################
    ### UNITTEST FEATURES

        # HIDDEN FEATURE -- CONVERT ANY INPUT TO A DATAFRAME
        feature_module = feature_options.get_tests('to_dataframe')
        num_tests_ran += run_tests(feature_module, yaml_config.debug.test_verbosity, {})

        # APPLY EACH NORMAL FEATURE
        for feature in yaml_config.features:
            feature_module = feature_options.get_tests(feature.name)
            num_tests_ran += run_tests(feature_module, yaml_config.debug.test_verbosity, {
                **feature.params,
                '_sample_dataset': sample_dataset
            })

        # HIDDEN FEATURE -- DROP ROWS WITH NANS
        feature_module = feature_options.get_tests('drop_nan_rows')
        num_tests_ran += run_tests(feature_module, yaml_config.debug.test_verbosity, {})

        # HIDDEN FEATURE -- EXTRACT FEATURE COLUMNS
        feature_module = feature_options.get_tests('extract_columns')
        num_tests_ran += run_tests(feature_module, yaml_config.debug.test_verbosity, { 'columns': yaml_config.training.feature_columns })

    #####################################################################################
    ### UNITTEST DATASET SEGMENTATION

        segmentation_module = segmentation_options.get_tests(yaml_config.training.segmentation.method)
        num_tests_ran += run_tests(segmentation_module, yaml_config.debug.test_verbosity, yaml_config.training.segmentation.params)

    #####################################################################################
    ### MAKE SURE ALL REQUIRED COLUMNS EXIST

        # FETCH BASELINE DATASET COLUMNS
        all_columns = list(yaml_config.dataset.expected_schema.keys())

        # APPEND IN EACH FEATURE COLUMN
        for feature in yaml_config.features:
            all_columns.append(feature.params['output_column'])

        # MAKE SURE LABEL COLUMN EXISTS
        label_error = f"LABEL COLUMN '{yaml_config.training.label_column}' DOES NOT EXIST.\nOPTIONS: {all_columns}"
        assert yaml_config.training.label_column in all_columns, label_error

        # MAKE SURE ALL FEATURE COLUMNS EXIST
        for column_name in yaml_config.training.feature_columns:
            feature_error = f"FEATURE COLUMN '{yaml_config.training.label_column}' DOES NOT EXIST.\nOPTIONS: {all_columns}"
            assert column_name in all_columns, feature_error

        # TODO: UNITTEST LABEL EXTRACTION FUNC
        # TODO: UNITTEST LABEL EXTRACTION FUNC
        # TODO: UNITTEST LABEL EXTRACTION FUNC

    #####################################################################################
    ### UNITTEST SCALER

        scaler_module = scaler_options.get_tests(yaml_config.training.scaler.name)
        num_tests_ran += run_tests(scaler_module, yaml_config.debug.test_verbosity, yaml_config.training.scaler.params)

    #####################################################################################
    ### UNITTEST MODEL

        model_ref: str = f'{yaml_config.training.model.type}.{yaml_config.training.model.name}'
        model_module = model_options.get_tests(model_ref)
        num_tests_ran += run_tests(model_module, yaml_config.debug.test_verbosity, yaml_config.training.model.params)

        # IF ALL TESTS PASSED, PROCEED WITH THE EXPERIMENT
        return True, num_tests_ran

    # OTHERWISE, AT LEAST ONE TEST FAILED
    # THEREFORE, BLOCK THE EXPERIMENT
    except AssertionError as error:
        print(f'\nINTERPRETER-SIDE ASSERTION ERROR:')
        print('----------------------------------------------------------------------')
        print(error)
        return False, num_tests_ran

# EXECUTE JUST THE PIPELINE TESTS
if __name__ == '__main__':
    raw_config: dict = files.load_yaml('pipeline.yaml')
    yaml_config = types.config_schema(**raw_config)
    run(yaml_config)