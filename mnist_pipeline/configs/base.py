from mnist_pipeline.dataset import MNISTImagesDataset, MNISTTargetsDataset

from pipeline.config_base import ConfigBase, PredictConfigBase
from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.datasets.base import DatasetWithPostprocessingFunc, DatasetComposer
from pipeline.trainers.classification import TrainerClassification
from pipeline.predictors.classification import PredictorClassification

import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor


TRAIN_DATASET_PATH = "~/.pipeline/mnist/train.csv"
TEST_DATASET_PATH = "~/.pipeline/mnist/test.csv"

VAL_RATIO = 0.2


def get_dataset(mode, transforms):
    images_dataset = DatasetWithPostprocessingFunc(
        MNISTImagesDataset(path=TRAIN_DATASET_PATH, mode=mode, val_ratio=VAL_RATIO),
        transforms)

    targets_dataset = MNISTTargetsDataset(
        path=TRAIN_DATASET_PATH, mode=mode, val_ratio=VAL_RATIO)

    return DatasetComposer([images_dataset, targets_dataset])


class ConfigMNISTBase(ConfigBase):
    def __init__(self, model, model_save_path, num_workers=4, batch_size=128, transforms=None):
        optimizer = optim.Adam(model.parameters())
        scheduler = SchedulerWrapperLossOnPlateau(optimizer)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()
        trainer_cls = TrainerClassification

        if transforms is None:
            transforms = ToTensor()

        train_dataset = get_dataset(mode=MNISTImagesDataset.MODE_TRAIN, transforms=transforms)
        val_dataset = get_dataset(mode=MNISTImagesDataset.MODE_VAL, transforms=transforms)

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


class PredictConfigMNISTBase(PredictConfigBase):
    def __init__(self, model, model_save_path, num_workers=4, batch_size=128):
        predictor_cls = PredictorClassification

        images_dataset = DatasetWithPostprocessingFunc(
            MNISTImagesDataset(path=TRAIN_DATASET_PATH, mode=MNISTImagesDataset.MODE_VAL, val_ratio=VAL_RATIO),
            ToTensor())

        dataset = DatasetComposer([images_dataset, list(range(len(images_dataset)))])

        super().__init__(
            model=model,
            model_save_path=model_save_path,
            dataset=dataset,
            predictor_cls=predictor_cls,
            num_workers=num_workers,
            batch_size=batch_size)
