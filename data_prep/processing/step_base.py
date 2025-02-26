import abc


class ProcessingStepBase:
    def __init__(self, cfg, step_cfg, state):
        pass

    @abc.abstractmethod
    def can_be_skipped(self, cfg, state):
        pass

    @abc.abstractmethod
    def run(self, cfg, state):
        pass

    @abc.abstractmethod
    def update_state(self, cfg, state, has_run):
        pass
