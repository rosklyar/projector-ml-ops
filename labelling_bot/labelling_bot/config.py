import json

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

config_path = 'config.json'
config = read_config(config_path)

LABELS = config['labels']
BUCKET_NAME = config['bucket_name']
ROOT_FOLDER = config['root_folder']
