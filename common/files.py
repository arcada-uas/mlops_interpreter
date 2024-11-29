import yaml, json, os, time

################################################################################################
################################################################################################

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=4) # allow_unicode=True

def save_json(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=False)

################################################################################################
################################################################################################

def logdump(pipeline_config: dict, result_overview: dict):

    # CREATE A NEW DIR DIRECTORY
    now = int(time.time())
    dir_path = f'runs/{now}'
    os.mkdir(dir_path)

    # DUMP PIPELINE & ITS RESULTS
    save_yaml(f'{dir_path}/pipeline.yaml', pipeline_config)
    save_json(f'{dir_path}/results.json', result_overview)