from mnist_pipeline.configs.simple_cnn import Config, PredictConfig

from pipeline.utils import run_train, run_predict
import tempfile
import shutil
import os
import hashlib


class TestMNISTTrain:
    def test_mnist_train(self):
        test_path = tempfile.mkdtemp()
        config = Config(model_save_path=test_path)
        config.epoch_count = 2
        run_train(config)

        assert os.path.exists(os.path.join(test_path, "log.txt"))
        assert os.path.exists(os.path.join(test_path, "epoch_0"))
        assert os.path.exists(os.path.join(test_path, "epoch_1"))
        assert not os.path.exists(os.path.join(test_path, "epoch_2"))
        assert os.path.exists(os.path.join(test_path, "state"))

        with open(os.path.join(test_path, "epoch_1"), "rb") as fin:
            model_checkpoint_hash = hashlib.md5(fin.read()).hexdigest()

        run_train(config)

        with open(os.path.join(test_path, "epoch_1"), "rb") as fin:
            new_model_checkpoint_hash = hashlib.md5(fin.read()).hexdigest()

        assert model_checkpoint_hash == new_model_checkpoint_hash
        assert not os.path.exists(os.path.join(test_path, "epoch_2"))

        predict_config = PredictConfig(model_save_path=test_path)
        run_predict(predict_config)

        assert os.path.exists(os.path.join(test_path, "predictions", "predictions"))
        assert os.path.exists(os.path.join(test_path, "predictions", "identifiers"))

        shutil.rmtree(test_path)
