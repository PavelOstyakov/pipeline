from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.core import PipelineError


import pytest


class TestClassificationMetrics:
    def test_accuracy(self):
        metrics_calculator = MetricsCalculatorAccuracy(border=0.4)

        with pytest.raises(PipelineError):
            metrics_calculator.calculate()

