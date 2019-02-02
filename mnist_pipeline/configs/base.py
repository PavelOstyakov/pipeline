from pipeline.config_base import ConfigBase
from pipeline.schedulers.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy

import torch.nn as nn
import torch.optim as optim


TRAIN_DATASET_PATH = "~/.pipeline/mnist/train.csv"
TEST_DATASET_PATH = "~/.pipeline/mnist/test.csv"


class ConfigMNISTBase(ConfigBase):
    def __init__(self, model):
        optimizer = optim.Adam(model.parameters())
        scheduler = SchedulerWrapperLossOnPlateau(optimizer)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            metrics_calculator=metrics_calculator,
            batch_size=32,
            num_workers=4
        )
