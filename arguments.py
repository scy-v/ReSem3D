"""load YAML config file"""
import os
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_config(config_path=None):
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_config(config_path)
    return config


