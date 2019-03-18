from imagenet_pipeline.dataset import ImageNetImagesDataset, ImageNetTargetsDataset

from pipeline.config_base import ConfigBase
from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.datasets.base import DatasetWithPostprocessingFunc, DatasetComposer, OneHotTargetsDataset
from pipeline.trainers.classification import TrainerClassification

from pipeline.datasets.mixup import MixUpDatasetWrapper
from pipeline.losses.vector_cross_entropy import VectorCrossEntropy
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor, Compose, Normalize


TRAIN_DATASET_PATH = "~/train"
TEST_DATASET_PATH = "~/val"


def get_dataset(path, transforms, use_mixup):
    images_dataset = DatasetWithPostprocessingFunc(
        ImageNetImagesDataset(path=path),
        transforms)

    targets_dataset = ImageNetTargetsDataset(path=path)

    if use_mixup:
        targets_dataset = OneHotTargetsDataset(targets_dataset, 1000)
    return DatasetComposer([images_dataset, targets_dataset])


class ConfigImageNetBase(ConfigBase):
    def __init__(self, model, model_save_path, num_workers=16, batch_size=128, learning_rate=0.1, transforms=None, use_mixup=False):
        parameters_bias = [p[1] for p in model.named_parameters() if 'bias' in p[0]]
        parameters_scale = [p[1] for p in model.named_parameters() if 'scale' in p[0]]
        parameters_others = [p[1] for p in model.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]

        optimizer = optim.SGD(
                    [{'params': parameters_bias, 'lr': learning_rate/10.},
                             {'params': parameters_scale, 'lr': learning_rate/10.},
                             {'params': parameters_others}],
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=5e-4)
        scheduler = SchedulerWrapperLossOnPlateau(optimizer)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()
        trainer_cls = TrainerClassification

        if transforms is None:
            transforms = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = get_dataset(path=TRAIN_DATASET_PATH, transforms=transforms, use_mixup=use_mixup)
        val_dataset = get_dataset(path=TEST_DATASET_PATH, transforms=transforms, use_mixup=use_mixup)
        
        if use_mixup:
            train_dataset = MixUpDatasetWrapper(train_dataset, alpha=0.7)
            loss = VectorCrossEntropy()

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
            trainer_cls=trainer_cls,
            print_frequency=100)
