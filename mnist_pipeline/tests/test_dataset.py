from mnist_pipeline.dataset import MNISTDataset, MNISTImagesDataset, MNISTTargetsDataset
from mnist_pipeline.configs.base import TRAIN_DATASET_PATH

from pipeline.utils import get_path

import os


class TestMNISTDataset:
    def setup(self):
        assert os.path.exists(get_path(TRAIN_DATASET_PATH)), "You need to download MNIST dataset first"

    def test_train_dataset(self):
        dataset = MNISTDataset(TRAIN_DATASET_PATH, mode=MNISTDataset.MODE_TRAIN, val_ratio=0.2)
        assert len(dataset) == 33600

        _, _ = dataset[33599]
        image, target = dataset[0]

        assert 0 <= target < 10

        assert image.shape == (1, 28, 28)

    def test_val_dataset(self):
        dataset = MNISTDataset(TRAIN_DATASET_PATH, mode=MNISTDataset.MODE_VAL, val_ratio=0.2)
        assert len(dataset) == 8400

        _, _ = dataset[8399]
        image, target = dataset[0]

        assert 0 <= target < 10

        assert image.shape == (1, 28, 28)

        dataset = MNISTDataset(TRAIN_DATASET_PATH, mode=MNISTDataset.MODE_VAL, val_ratio=0)
        assert len(dataset) == 0

    def test_images_dataset(self):
        dataset = MNISTImagesDataset(TRAIN_DATASET_PATH, mode=MNISTDataset.MODE_VAL, val_ratio=1)

        image = dataset[10]
        assert image.shape == (1, 28, 28)

        assert image.min() >= 0
        assert 1 <= image.max() <= 255

    def test_targets_dataset(self):
        dataset = MNISTTargetsDataset(TRAIN_DATASET_PATH, mode=MNISTDataset.MODE_TRAIN, val_ratio=0.5234)

        target = dataset[51]

        assert 0 <= target <= 9

        assert type(target) == int
