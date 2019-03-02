from imagenet_pipeline.dataset import ImageNetImagesDataset, ImageNetTargetsDataset

from pipeline.config_base import ConfigBase
from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.datasets.base import DatasetWithPostprocessingFunc, DatasetComposer
from pipeline.trainers.classification import TrainerClassification

import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor


TRAIN_DATASET_PATH = "~/.pipeline/imagenet/train"
TEST_DATASET_PATH = "~/.pipeline/imagenet/val"


def get_dataset(path, transforms):
    images_dataset = DatasetWithPostprocessingFunc(
        ImageNetImagesDataset(path=path),
        transforms)

    targets_dataset = ImageNetTargetsDataset(path=path)

    return DatasetComposer([images_dataset, targets_dataset])


class ConfigImageNetBase(ConfigBase):
    def __init__(self, model, model_save_path, num_workers=8, batch_size=128, transforms=None):
        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)#, weight_decay=5e-4)
        scheduler = SchedulerWrapperLossOnPlateau(optimizer)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()
        trainer_cls = TrainerClassification

        if transforms is None:
            transforms = ToTensor()

        train_dataset = get_dataset(path=TRAIN_DATASET_PATH, transforms=transforms)
        val_dataset = get_dataset(path=TEST_DATASET_PATH, transforms=transforms)

        super().__init__(
            model=model,
            model_save_path=model_save_path,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            metrics_calculator=metrics_calculator,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            trainer_cls=trainer_cls)
