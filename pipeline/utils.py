from .logger import setup_logger

from torch.utils.data import DataLoader
from torch.nn import DataParallel

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


def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    with open(path, "wb") as fout:
        torch.save(model.state_dict(), fout)


def load_model(model, path):
    with open(path, "rb") as fin:
        state_dict = torch.load(fin)

    model.load_state_dict(state_dict)


def run_train(config):
    train_data_loader = DataLoader(
        config.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers)

    val_data_loader = DataLoader(
        config.val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    model = config.model

    model_save_path = config.model_save_path
    os.makedirs(model_save_path, exist_ok=True)

    logger_path = os.path.join(model_save_path, "log.txt")
    setup_logger(out_file=logger_path)

    trainer = config.trainer_cls(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        epoch_count=config.epoch_count,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        loss=config.loss,
        metrics_calculator=config.metrics_calculator,
        print_frequency=config.print_frequency,
        device=config.device,
        model_save_path=config.model_save_path,
        state_storage=config.state_storage
    )

    trainer.run()
