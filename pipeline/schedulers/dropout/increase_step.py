from ..base import SchedulerBase

from .utils import set_dropout_probability


class SchedulerWrapperIncreaseStep(SchedulerBase):
    def __init__(self, model, epoch_count, initial_value=0, max_value=0.5):
        self._model = model
        self._epoch_count = epoch_count
        self._initial_value = initial_value
        self._max_value = max_value

    def step(self, loss, metrics, epoch_id):
        new_value = (self._max_value - self._initial_value) / self._epoch_count * (epoch_id + 1)
        set_dropout_probability(self._model, new_value)
