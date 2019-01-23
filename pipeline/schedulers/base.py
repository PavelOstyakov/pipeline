import abc


class SchedulerWrapperBase(abc.ABC):
    def __init__(self, scheduler):
        self._scheduler = scheduler

    @abc.abstractmethod
    def step(self, loss, metrics, epoch_id):
        pass


class SchedulerWrapperIdentity(SchedulerWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(None)

    def step(self, loss, metrics, epoch_id):
        pass


class SchedulerWrapperLossBase(SchedulerWrapperBase):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, loss, metrics, epoch_id):
        return self._scheduler.step(loss, epoch_id)


class SchedulerWrapperMetricsMeanBase(SchedulerWrapperBase):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, loss, metrics, epoch_id):
        values = list(metrics.values())
        mean_metrics = sum(values) / len(values)
        return self._scheduler.step(mean_metrics, epoch_id)