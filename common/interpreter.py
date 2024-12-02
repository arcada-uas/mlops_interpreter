from common import pydantic, misc

################################################################################################
################################################################################################

class create_repository:
    def __init__(self, options_mapping: dict, label: str):
        self.options = options_mapping
        self.keys = list(options_mapping.keys())
        self.label = label.upper()

    # FETCH OPTION
    def get(self, option_name: str):
        assert isinstance(option_name, str), f"ARG '{option_name}' MUST BE OF TYPE STR, GOT {type(option_name)}"
        assert option_name in self.keys, f"{self.label} '{option_name}' WAS NOT FOUND\nOPTIONS: {self.keys}"

        return self.options[option_name].module
    
    # FETCH & INSTANTIATE OPTION
    def create(self, option_name: str, option_params: dict = {}):
        assert isinstance(option_name, str), f"ARG '{option_name}' MUST BE OF TYPE STR, GOT {type(option_name)}"
        assert isinstance(option_params, dict), f"ARG '{option_params}' MUST BE OF TYPE DICT, GOT {type(option_params)}"

        # WITH PARAMS
        if len(option_params) > 0:
            return self.get(option_name)(**option_params)

        # WITHOUT PARAMS
        return self.get(option_name)()
    
    # FETCH UNITTESTS FOR OPTION
    def get_tests(self, option_name: str):
        assert isinstance(option_name, str), f"ARG '{option_name}' MUST BE OF TYPE STR, GOT {type(option_name)}"
        assert option_name in self.keys, f"{self.label} '{option_name}' WAS NOT FOUND\nOPTIONS: {self.keys}"

        return self.options[option_name].tests
    
class option:
    def __init__(self, module, tests):
        self.module = module
        self.tests = tests

################################################################################################
################################################################################################

def handle_errors(error):

    # BASIC ASSERT ERRORS
    if type(error) == AssertionError:
        misc.print_header('INTERPRETER-SIDE ASSERTION ERROR:')
        return print(error)

    # PARSE PYDANTIC ERRORS
    if type(error) == pydantic.ValidationError:
        prettified_error = pydantic.parse_pydantic_error(error)
        return print(prettified_error)
    
    # RENDER NOTHING EXTRA FOR INTENTIONALLY THROWN UNITTEST ERROR
    if str(error) == 'UNITTEST_ERROR':
        return
    
    # OTHER ERRORS
    misc.print_header('INTERPRETER-SIDE FATAL ERROR:')
    print(error)