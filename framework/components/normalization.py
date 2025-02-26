import abc
from typing import Callable
import torch
import os

from framework.cache_manager import CacheDir
from framework.logger import logger
import framework.util.file_utils as file_utils


class BaseNormalization:
    def __init__(self, cfgs) -> None:
        super().__init__()
        self.cfgs = cfgs
        self.cache = CacheDir(cfgs)
        self.norm_params = None

    def initialize(self, data: list, combined_data: dict):
        if self._has_already_been_cached():
            self.norm_params = self._load_from_cache()
        else:
            self.norm_params = self._calculate_normalization_params(data, combined_data)
            self._save_normalization_params_to_disk(self.norm_params)

    def normalize(self, data):
        assert (
            self.norm_params is not None
        ), "Trying to call normalization component without initialization"
        for idx, entry in enumerate(data):
            data[idx] = self.normalize_single(entry)
        return data

    def _save_normalization_params_to_disk(self, norm_params: dict):
        file_utils.write_dict_to_json(norm_params, self.cache_fp)

    def _load_from_cache(self) -> dict:
        return file_utils.read_dict_from_json(self.cache_fp)

    def _has_already_been_cached(self) -> bool:
        return self.cache.exists(self.cache_name) and os.path.exists(self.cache_fp)

    @abc.abstractmethod
    def normalize_single(self, item: dict) -> dict:
        pass

    @abc.abstractmethod
    def normalize_xyz(self, xyz: torch.tensor):
        pass

    @abc.abstractmethod
    def denormalize(self, item: dict) -> torch.tensor:
        pass

    @abc.abstractmethod
    def _calculate_normalization_params(self, data: list, combined_data: dict) -> dict:
        pass
