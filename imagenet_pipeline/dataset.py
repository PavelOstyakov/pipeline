from pipeline.core import PipelineError
from pipeline.utils import get_path

from PIL import Image

import torch.utils.data as data

import os
import glob

IMAGE_SIZE = (224, 224)


class ImageNetDataset(data.Dataset):
    def __init__(self, path):
        path = get_path(path)
        if not os.path.exists(path):
            raise PipelineError("Path {} does not exist".format(path))

        self._paths = sorted(glob.glob(os.path.join(path, "*/*.JPEG")))

        classes = set()
        for path in self._paths:
            class_name = os.path.basename(os.path.dirname(path))
            classes.add(class_name)

        classes = sorted(list(classes))
        self._class_to_id = dict((class_name, i) for i, class_name in enumerate(classes))

    def get_image(self, item):
        path = self._paths[item]
        image = Image.open(path).resize(IMAGE_SIZE).convert("RGB")
        return image

    def get_class(self, item):
        path = self._paths[item]
        class_name = os.path.basename(os.path.dirname(path))
        result = self._class_to_id[class_name]
        return result

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        return self.get_image(item), self.get_class(item)


class ImageNetImagesDataset(ImageNetDataset):
    def __getitem__(self, item):
        return self.get_image(item)


class ImageNetTargetsDataset(ImageNetDataset):
    def __getitem__(self, item):
        return self.get_class(item)
