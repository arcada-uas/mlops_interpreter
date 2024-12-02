from common import file_utils, types, testing, interpreter, misc
from pandas import DataFrame
import coverage

from mappings.data_retrieval import repository as retrieval_options
from mappings.features import repository as feature_options
from mappings.models import repository as model_options
from mappings.scalers import repository as scaler_options
from mappings.segmentation import repository as segmentation_options

##############################################################################################################
##############################################################################################################

def run(yaml_config: types.config_schema):
    try:
        misc.print_header('UNITTESTING PIPELINE', init_space=False)

        # START UNITTEST COVERAGE MODULE
        # IGNORE FILES FROM COMMON
        cov = coverage.Coverage(data_file=None, omit='common/*')
        cov.start()

        # KEEP TRACK OF HOW MANY TESTS HAVE BEEN RAN
        num_tests_ran: int = 0

    #####################################################################################
    ### UNITTEST DATASET RETRIEVAL

        dataset_module = retrieval_options.get_tests(yaml_config.dataset.method)
        num_tests_ran += testing.run_tests(dataset_module, yaml_config.debug.test_verbosity, yaml_config.dataset.params)

        # THE TESTS PASSED -- LOAD IN A SAMPLE OF THE REAL DATASET FOR OTHER TESTS
        fetching_func = retrieval_options.get(yaml_config.dataset.method)
        sample_dataset = DataFrame(fetching_func(**yaml_config.dataset.params, unittest_limit=200))

        # SELECT A SAMPLE ROW & MAKE SURE IT FOLLOWS EXPECTED SCHEMA
        sample_row = sample_dataset.iloc[0].to_dict()
        testing.validate_expected_schema(sample_row, yaml_config.dataset.expected_schema)

    #####################################################################################
    ### UNITTEST FEATURES

        # HIDDEN FEATURE -- CONVERT ANY INPUT TO A DATAFRAME
        feature_module = feature_options.get_tests('to_dataframe')
        num_tests_ran += testing.run_tests(feature_module, yaml_config.debug.test_verbosity, {})

        # APPLY EACH NORMAL FEATURE
        for feature in yaml_config.features:
            feature_module = feature_options.get_tests(feature.method)
            num_tests_ran += testing.run_tests(feature_module, yaml_config.debug.test_verbosity, {
                **feature.params,
                '_sample_dataset': sample_dataset
            })

        # HIDDEN FEATURE -- DROP ROWS WITH NANS
        feature_module = feature_options.get_tests('drop_nan_rows')
        num_tests_ran += testing.run_tests(feature_module, yaml_config.debug.test_verbosity, {})

        # HIDDEN FEATURE -- EXTRACT FEATURE COLUMNS
        feature_module = feature_options.get_tests('extract_features')
        num_tests_ran += testing.run_tests(feature_module, yaml_config.debug.test_verbosity, {
            'columns': yaml_config.training.feature_columns
        })

    #####################################################################################
    ### UNITTEST DATASET SEGMENTATION

        segmentation_module = segmentation_options.get_tests(yaml_config.training.segmentation.method)
        num_tests_ran += testing.run_tests(segmentation_module, yaml_config.debug.test_verbosity, yaml_config.training.segmentation.params)

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

        scaler_module = scaler_options.get_tests(yaml_config.training.scaler.method)
        num_tests_ran += testing.run_tests(scaler_module, yaml_config.debug.test_verbosity, yaml_config.training.scaler.params)

    #####################################################################################
    ### UNITTEST MODEL

        model_ref: str = f'{yaml_config.training.model.type}.{yaml_config.training.model.method}'
        model_module = model_options.get_tests(model_ref)
        num_tests_ran += testing.run_tests(model_module, yaml_config.debug.test_verbosity, yaml_config.training.model.params)

        # STOP UNITTEST COVERAGE MODULE
        cov.stop()

        # PRINT UNITTEST COVERAGE REPORT
        misc.clear_console()
        misc.print_header(f'{num_tests_ran}/{num_tests_ran} UNITTESTS PASSED -- SHOWING COVERAGE', init_space=False)
        cov.report()

        return True, num_tests_ran

    except Exception as error:
        interpreter.handle_errors(error)
        return False, None

##############################################################################################################
##############################################################################################################

# EXECUTE JUST THE PIPELINE TESTS
if __name__ == '__main__':
    raw_config: dict = file_utils.load_yaml('pipeline.yaml')
    yaml_config = types.config_schema(**raw_config)
    run(yaml_config)