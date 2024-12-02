from pydantic import BaseModel, Field, ValidationError
import json

##############################################################################################################
##############################################################################################################

# CHANGE SOME FUNCTIONALITY FOR THE BASE SCHEMA THAT ALL OTHER SCHEMAS USE
class base_schema(BaseModel):
    def __init__(self, *args, **kwargs):

        # ENABLE USAGE my_schema(foo, bar, biz)
        # RATHER THAN THE REPETITIVE my_schema(foo=foo, bar=bar, biz=biz)
        # STILL ALLOWS FOR my_schema(**my_params) WITH DICTS
        field_names = list(self.__annotations__.keys())
    
        if args:
            assert len(args) == len(field_names), f"Expected {len(field_names)} positional arguments, got {len(args)}"
            kwargs.update(zip(field_names, args))

        super().__init__(**kwargs)

##############################################################################################################
##############################################################################################################

def parse_pydantic_error(pydantic_error: ValidationError):
    schema_name = pydantic_error.title
    parsed_errors = json.loads(pydantic_error.json())
    error_strings = []

    for error in parsed_errors:
        variable_name = ', '.join(error['loc'])
        # msg = f"'{schema_name}.{variable_name}' -> {error['msg']} (found {error['input']})"
        msg = f"'{schema_name}.{variable_name}' -> {error['msg']}"
        error_strings.append(msg)
        
    return '\n'.join(error_strings)