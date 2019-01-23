import abc


class MetricsCalculatorBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def zero_cache(self):
        pass

    @abc.abstractmethod
    def add(self, y_predicted, y_true):
        pass

    @abc.abstractmethod
    def calculate(self):
        pass


class MetricsCalculatorEmpty(MetricsCalculatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def zero_cache(self):
        pass

    def add(self, y_predicted, y_true):
        pass

    def calculate(self):
        return {}
