from abc import ABC
from pandas import DataFrame
from common import pydantic, misc
from sklearn.datasets import make_regression
import unittest, os, json, time

# THE ENVIRONMENT VAR THAT UNITTESTS REQUIRE TO OBTAIN DYNAMIC ARGUMENTS
env_var_name: str = '_UNITTEST_ARGS'

################################################################################################
################################################################################################

# SHARED CONSTRUCTOR FOR ALL UNITTESTS
class base_unittest(unittest.TestCase, ABC):
    def setUp(self):

        # SAVE MODULE YAML PARAMS IN STATE
        stringified_dict: str = os.environ.get(env_var_name)
        parsed_params: dict = json.loads(stringified_dict)

        # EXTRACT & SAVE THE SAMPLE DATASET WHEN ONE WAS PROVIDED
        if '_sample_dataset' in parsed_params:
            self.sample_dataset = DataFrame(parsed_params['_sample_dataset'])
            del parsed_params['_sample_dataset']

        # SAVE THE REMAINING PARAMS IN STATE
        self.yaml_params: dict = parsed_params

################################################################################################
################################################################################################

class StopOnFirstErrorResult(unittest.TextTestResult):
    def addError(self, test, err):
        exc_type, exc_value, something = err

        # OVERRIDE UGLY PYDANTIC ERRORS
        if exc_type == pydantic.ValidationError:
            misc.clear_console()
            pretty_error = pydantic.parse_pydantic_error(exc_value)
            err = (AssertionError, AssertionError(pretty_error), something)
         
        misc.hide_traces()
        super().addError(test, err)
        self.stop()

    def addFailure(self, test, err):
        exc_type, exc_value, something = err

        # OVERRIDE UGLY PYDANTIC ERRORS
        if exc_type == pydantic.ValidationError:
            misc.clear_console()
            pretty_error =  pydantic.parse_pydantic_error(exc_value)
            err = (AssertionError, AssertionError(pretty_error), something)

        misc.hide_traces()
        super().addFailure(test, err)
        self.stop()

################################################################################################
################################################################################################

def run_tests(module, verbosity_level: int, yaml_params: dict):

    # HANDLE SAMPLE DATASET FORMATTING
    if '_sample_dataset' in yaml_params:
        yaml_params['_sample_dataset'] = yaml_params['_sample_dataset'].to_dict(orient='records')

    # MAKE INPUT ARGS AVAILABLE FOR THE UNITTESTS THROUGH ENVIRONMENT
    os.environ[env_var_name] = json.dumps(yaml_params)

    # CREATE THE TESTING SUITE
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(module))
    
    # RAISE ERROR IF NO UNITTESTS WERE FOUND
    if len(suite._tests) == 0:
        raise Exception(f"MODULE '{module}' HAS NO UNITTESTS")
    
    # OTHERWISE, RUN THE TESTS
    runner = unittest.TextTestRunner(verbosity=verbosity_level, resultclass=StopOnFirstErrorResult)
    output = runner.run(suite)

    # KILL PARENT PROCESS IF YOU FIND ANY ERRORS/FAILS
    if len(output.errors) > 0 or len(output.failures) > 0:
        raise Exception('UNITTEST_ERROR')     

    # OTHERWISE, RETURN THE NUMBER OF TESTS THAT WERE RAN 
    return output.testsRun

################################################################################################
################################################################################################

def validate_expected_schema(sample_row: dict, expected_schema: dict):
    reference_schema = {}

    # YAML REFERS TO TYPES BY STRING NAME
    type_mapping = {
        'str': str,
        'int': int,
        'float': float,
    }

    # BUILD THE REFERENCE SCHEMA
    # SWAP STRINGIFIED TYPES WITH REAL TYPES
    for key, key_type in expected_schema.items():
        key_error = f"TYPE '{key_type}' FOR DATASET COLUMN '{key}' MISSING FROM UNITTEST TYPE MAPPING\nSOLUTION: IF THE TYPE IS CORRECT, ADD IT MANUALLY"
        assert key_type in type_mapping, key_error
        reference_schema[key] = type_mapping[key_type]

    for key in reference_schema.keys():

        # MAKE SURE THE COLUMN EXISTS
        key_error: str = f"DATASET COLUMN '{key}' NOT FOUND IN DATASET"
        assert key in sample_row, key_error

        value_type = type(sample_row[key])
        expected_type = reference_schema[key]

        # MAKE SURE ITS OF THE CORRECT TYPE
        value_error: str = f"DATASET COLUMN '{key}' IS OF WRONG TYPE\nEXPECTED {expected_type}, FOUND {value_type}"
        assert value_type == expected_type, value_error

################################################################################################
################################################################################################

class create_synth_dataset_schema(pydantic.BaseModel):
    column_names: list[str] = pydantic.Field(min_length=1)
    num_rows: int = pydantic.Field(ge=1)
    random_state: int

def create_synth_dataset(column_names: list[str], num_rows: int, random_state: int, columns_only: bool = False, as_df: bool = True):
    params = create_synth_dataset_schema(
        column_names=column_names,
        num_rows=num_rows,
        random_state=random_state,
    )

    # GENERATE A FEATURE MATRIX
    float_matrix, _ = make_regression(
        n_samples=params.num_rows,
        n_features=len(params.column_names),
        random_state=params.random_state,
        noise=10.0,
    )

    # GENERATE A STARTING UNIX TIMESTAMP
    start_time = int(time.time())

    # STITCH TOGETHER FEATURES & COLUMN NAMES
    dataset = [{
        'symbol': 'SYNTH',
        'timestamp': start_time + nth,
        **dict(zip(params.column_names, row_features)),
    } for nth, row_features in enumerate(float_matrix)]

    # WHEN REQUESTED, LIST OF DICTS
    if as_df is not True:
        return dataset
    
    # OTHERWISE, CONVERT TO DATAFRAME
    dataset = DataFrame(dataset)

    # WHEN REQUESTED, ONLY SPECIFIC COLUMNS
    if columns_only is True:
        return dataset[column_names]
    
    return dataset