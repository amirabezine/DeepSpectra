import os
import yaml

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_config():
    config_path = os.path.join(get_project_root(), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resolve_path(path):
    return os.path.join(get_project_root(), path)
