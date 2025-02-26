import abc
import torch


class BaseTrainingStep:
    @abc.abstractmethod
    def training_step(self, pipeline, batch: torch.tensor, batch_idx: torch.tensor):
        pass

    def after_training_step(self, pipeline, outputs, batch, batch_idx):
        pass
