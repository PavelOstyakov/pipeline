import time
from typing import Iterable

import torch
import torch.nn as nn

from ..logger import LOGGER
from ..storage.predictions import PredictionsStorageBase
from ..utils import move_to_device, load_model

import os


class PredictorBase:
    def __init__(
            self,
            model: nn.Module,
            data_loader: Iterable,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            predictions_storage: PredictionsStorageBase) -> None:

        self.model = model.to(device)
        self.data_loader = data_loader
        self.print_frequency = print_frequency
        self.device = device
        self.model_save_path = model_save_path
        self.predictions_storage = predictions_storage

    def predict_step(self, input_data: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)
        model_output = self.model(input_data)
        return model_output

    def log_predict_step(self, step_id: int, predict_time: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Predict step {}".format(predict_time, step_id))
            return True

        return False

    def log_predict_completed(self, predict_time: float):
        LOGGER.info("[{} s] Predict is completed".format(predict_time))
        return True

    def load_last_model(self):
        if os.path.exists(self.model_save_path):
            epochs = filter(lambda file: file.startswith("epoch_"), os.listdir(self.model_save_path))
            epochs = map(lambda file: int(file[file.find("_") + 1]), epochs)
            epochs = list(epochs)

            if epochs:
                last_model_path = os.path.join(self.model_save_path, "epoch_{}".format(max(epochs)))
                load_model(self.model, last_model_path)
                return

        LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))

    def run(self):
        self.load_last_model()
        self.model.eval()

        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data, ids) in enumerate(self.data_loader):
                model_output = self.predict_step(input_data)
                self.predictions_storage.add_batch(ids, model_output)

                step_count += 1
                predict_time = time.time() - start_time
                self.log_predict_step(step_id, predict_time)

        self.predictions_storage.sort_by_id()
        self.predictions_storage.flush()
        predict_time = time.time() - start_time
        self.log_predict_completed(predict_time)
        return predict_time
