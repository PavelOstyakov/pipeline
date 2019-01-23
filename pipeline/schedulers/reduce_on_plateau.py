from .base import SchedulerWrapperLossBase, SchedulerWrapperMetricsMeanBase

from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerWrapperLossOnPlateau(SchedulerWrapperLossBase):
    def __init__(self, optimizer, mode="min", factor=0.5, patience=3, verbose=True, cooldown=3, min_lr=1e-8):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            cooldown=cooldown,
            min_lr=min_lr
        )
        super().__init__(scheduler)


class SchedulerWrapperMetricsMeanOnPlateau(SchedulerWrapperMetricsMeanBase):
    def __init__(self, optimizer, mode="max", factor=0.5, patience=3, verbose=True, cooldown=3, min_lr=1e-8):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            cooldown=cooldown,
            min_lr=min_lr
        )
        super().__init__(scheduler)
