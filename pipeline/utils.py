import importlib
import torch
import os


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    assert module.hasattr("Config"), "Config file should contain Config class"

    config = module.Config()
    return config


def move_to_device(tensor: list or tuple or torch.Tensor, device: str):
    if isinstance(tensor, list):
        return [move_to_device(elem, device=device) for elem in tensor]
    if isinstance(tensor, tuple):
        return (move_to_device(elem, device=device) for elem in tensor)
    return tensor.to(device)


def get_path(path):
    return os.path.expanduser(path)
