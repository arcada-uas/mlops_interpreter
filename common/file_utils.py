import yaml, json, joblib

################################################################################################
################################################################################################

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(file_path: str, data: dict):
    assert file_path.endswith('.yaml'), 'YAML FILENAME SHOULD END WITH .yaml'
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=4) # allow_unicode=True

################################################################################################
################################################################################################

def load_json(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(file_path: str, data: dict):
    assert file_path.endswith('.json'), 'JSON FILENAME SHOULD END WITH .json'
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=False)

################################################################################################
################################################################################################

def load_pickle(file_path: str):
    return joblib.load(file_path)

def save_pickle(file_path: str, object):
    assert file_path.endswith('.pkl'), 'PICKLE FILENAME SHOULD END WITH .pkl'
    joblib.dump(object, file_path)