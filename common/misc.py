from datetime import datetime
import json, os, platform, sys

################################################################################################
################################################################################################

def unix_ts(date_string: str) -> int:
    date_format = '%Y-%m-%d %H:%M:%S'
    datetime_obj = datetime.strptime(date_string, date_format)
    return int(datetime_obj.timestamp())

################################################################################################
################################################################################################

def pprint(data_dict: dict):
    print(json.dumps(data_dict, indent=4))

def print_header(header: str, init_space=True):
    if init_space: print()
    print(f'{header}')
    print('--------------------------------------------------------------------')

def formatted_print(items):
    max_left_width = 47
    max_right_width = 20
    left, right = items

    left_str = str(left)
    right_str = str(right)
    left_spaces = " " * (max_left_width - len(left_str))
    right_str = right_str.rjust(max_right_width)
    
    print(f"{left_str}:{left_spaces}{right_str}")

################################################################################################
################################################################################################

def clear_console():
    if platform.system() == "Windows":
        return os.system("cls")
    
    os.system("clear")

def hide_traces():
    sys.tracebacklimit = 0

################################################################################################
################################################################################################

def transpose_dict(data_dict):
    output = {}

    # FISH OUT THE LIST OF METRICS
    first_segment_name = list(data_dict.keys())[0]
    metric_names = list(data_dict[first_segment_name].keys())

    # TRANSPOSE METRICS FOR READABILITY
    for metric_name in metric_names:
        container = {}

        for segment_name, segment_metrics in data_dict.items():
            container[segment_name] = segment_metrics[metric_name]

        output[metric_name] = container

    return output

################################################################################################
################################################################################################