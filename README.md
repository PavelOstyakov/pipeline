# Pipeline

## How to run training

First of all, create a config. You may find some examples of configs in folders mnist_pipeline, cifar_pipeline and imagenet_pipeline.
Then, call:

`python3 bin/train.py path_to_config`


For example, for reproducing results from Fixup paper (actually, they are not reproducible) just call:

`python3 bin/train.py cifar_pipeline/configs/resnet110_fixup.py`
