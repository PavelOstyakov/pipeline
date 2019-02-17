from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau, SchedulerWrapperMetricsMeanOnPlateau

from torch.optim import Adam
import torch.nn as nn


class TestReduceLROnPlateau:
    def test_wrapper_loss(self):
        first_layer = nn.Linear(10, 5)
        second_layer = nn.Linear(5, 1)

        optimizer = Adam([{"params": first_layer.parameters(), "lr": 1},
                          {"params": second_layer.parameters(), "lr": 2}])
        scheduler = SchedulerWrapperLossOnPlateau(optimizer, factor=0.5, patience=1, min_lr=0.1, cooldown=2)

        assert optimizer.param_groups[0]["lr"] == 1
        assert optimizer.param_groups[1]["lr"] == 2

        scheduler.step(loss=10, metrics={"a": 5}, epoch_id=0)
        assert optimizer.param_groups[0]["lr"] == 1
        assert optimizer.param_groups[1]["lr"] == 2

        scheduler.step(loss=11, metrics={"a": 3}, epoch_id=1)
        assert optimizer.param_groups[0]["lr"] == 1
        assert optimizer.param_groups[1]["lr"] == 2

        scheduler.step(loss=12, metrics={"a": 1}, epoch_id=2)
        assert optimizer.param_groups[0]["lr"] == 0.5
        assert optimizer.param_groups[1]["lr"] == 1

        scheduler.step(loss=13, metrics={"a": 2}, epoch_id=3)
        scheduler.step(loss=14, metrics={"a": 5}, epoch_id=4)
        scheduler.step(loss=14, metrics={"a": 2}, epoch_id=5)

        assert optimizer.param_groups[0]["lr"] == 0.5
        assert optimizer.param_groups[1]["lr"] == 1

        scheduler.step(loss=14, metrics={"a": 100}, epoch_id=6)
        assert optimizer.param_groups[0]["lr"] == 0.25
        assert optimizer.param_groups[1]["lr"] == 0.5

        scheduler.step(loss=9, metrics={"a": 21}, epoch_id=7)
        scheduler.step(loss=8, metrics={"a": 21}, epoch_id=7)

        assert optimizer.param_groups[0]["lr"] == 0.25
        assert optimizer.param_groups[1]["lr"] == 0.5

        scheduler.step(loss=13, metrics={"a": 3}, epoch_id=8)

        assert optimizer.param_groups[0]["lr"] == 0.25
        assert optimizer.param_groups[1]["lr"] == 0.5

        scheduler.step(loss=14, metrics=None, epoch_id=9)

        assert optimizer.param_groups[0]["lr"] == 0.125
        assert optimizer.param_groups[1]["lr"] == 0.25

        for epoch_id in range(10, 30):
            scheduler.step(loss=14, metrics={"absd": "asdasd"}, epoch_id=epoch_id)

        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[1]["lr"] == 0.1

    def test_wrapper_metrics(self):
        model = nn.Linear(10, 1)

        optimizer = Adam(model.parameters(), lr=1)
        scheduler = SchedulerWrapperMetricsMeanOnPlateau(optimizer, factor=0.5, patience=0, min_lr=0.1, cooldown=0)

        assert optimizer.param_groups[0]["lr"] == 1

        scheduler.step(loss=None, metrics={"a": 1, "b": 1}, epoch_id=0)
        assert optimizer.param_groups[0]["lr"] == 1

        scheduler.step(loss="abacaba", metrics={"a": 1, "b": 0}, epoch_id=1)
        scheduler.step(loss=-10, metrics={"a": 1, "b": 1}, epoch_id=2)
        assert optimizer.param_groups[0]["lr"] == 0.25

        scheduler.step(loss=123, metrics={"a": 1, "b": 2}, epoch_id=3)
        assert optimizer.param_groups[0]["lr"] == 0.25

        scheduler.step(loss=0, metrics={"a": 2}, epoch_id=4)
        assert optimizer.param_groups[0]["lr"] == 0.25

        scheduler.step(loss=0, metrics={"aasda": 1.1}, epoch_id=5)
        assert optimizer.param_groups[0]["lr"] == 0.125

        for epoch_id in range(6, 20):
            scheduler.step(loss=0, metrics={"c": 1}, epoch_id=epoch_id)
            assert optimizer.param_groups[0]["lr"] == 0.1
