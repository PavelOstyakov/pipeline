from .base import SchedulerWrapperLossBase, SchedulerWrapperMetricsMeanBase

from torch.optim.lr_scheduler import CosineAnnealingLR


class SchedulerWrapperLossOnCyclic(SchedulerWrapperLossBase):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(scheduler)


class SchedulerWrapperMetricsMeanOnCyclic(SchedulerWrapperMetricsMeanBase):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(scheduler)