import importlib


def load_config(config_path):
    module = importlib.import_module(config_path)
    assert module.hasattr("Config"), "Config file should contain Config class"

    config = module.Config()
    return config
