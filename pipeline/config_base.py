from .datasets.base import EmptyDataset
from .metrics.base import MetricsCalculatorEmpty
from .schedulers.base import SchedulerWrapperIdentity


class ConfigBase:
    def __init__(self, model=None, train_dataset=None, val_dataset=None,
                 optimizer=None, scheduler=None, loss=None, metrics_calculator=None,
                 batch_size=1, num_workers=0):
        assert model is not None, "Please, specify a correct model in your config"
        assert train_dataset is not None, "Please, specify a correct train_dataset in your config"
        assert optimizer is not None, "Please, specify a correct optimizer in your config"
        assert loss is not None, "Please, specify a correct loss in your config"

        if val_dataset is None:
            val_dataset = EmptyDataset()

        if scheduler is None:
            scheduler = SchedulerWrapperIdentity()

        if metrics_calculator is None:
            metrics_calculator = MetricsCalculatorEmpty()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.metrics_calculator = metrics_calculator
        self.loss = loss
        self.optimizer = optimizer

