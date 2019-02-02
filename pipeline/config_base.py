from .datasets.base import EmptyDataset
from .metrics.base import MetricsCalculatorEmpty
from .schedulers.base import SchedulerWrapperIdentity
from .storage.state import StateStorageFile

import torch
import os


class ConfigBase:
    def __init__(
            self,
            model,
            model_save_path,
            train_dataset,
            optimizer,
            loss,
            trainer_cls,
            device=None,
            val_dataset=None,
            scheduler=None,
            metrics_calculator=None,
            batch_size=1,
            num_workers=0,
            epoch_count=None,
            print_frequency=1,
            state_storage=None):

        if val_dataset is None:
            val_dataset = EmptyDataset()

        if scheduler is None:
            scheduler = SchedulerWrapperIdentity()

        if metrics_calculator is None:
            metrics_calculator = MetricsCalculatorEmpty()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if state_storage is None:
            state_storage = StateStorageFile(os.path.join(model_save_path, "state"))

        self.model = model
        self.model_save_path = model_save_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.metrics_calculator = metrics_calculator
        self.loss = loss
        self.optimizer = optimizer
        self.epoch_count = epoch_count
        self.print_frequency = print_frequency
        self.trainer_cls = trainer_cls
        self.device = device
        self.state_storage = state_storage
