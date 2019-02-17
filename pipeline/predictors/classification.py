from .base import PredictorBase
import torch


class PredictorClassification(PredictorBase):
    def predict_step(self, input_data: torch.Tensor):
        result = super().predict_step(input_data)
        result = torch.softmax(result, dim=-1)
        return result
