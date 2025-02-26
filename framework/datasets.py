import abc
import json
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np

from framework.cache_manager import CacheDir
from framework.util.file_utils import read_dict_from_json
from framework.components.normalization import BaseNormalization
from framework.components.coordinate_systems import BaseCoordinateSystem
from framework.logger import logger
import framework.util.file_utils as file_utils


class BaseDataset(Dataset):
    def __init__(self, cfgs, dataset_name: str, split: str) -> None:
        super().__init__()
        self.cfgs = cfgs
        self.split = split
        self.load_as_if_test = False
        self.dataset_name = f"{dataset_name}_{split}"
        self.cache = CacheDir(cfgs)
        # read in root.json info file about dataset
        self.root = read_dict_from_json(
            os.path.join(
                cfgs.run.dataset_dp,
                "root.json",
            )
        )
        self.aoi_name = self.root.get("aoi_name")
        self.img_dp = os.path.join(cfgs.run.dataset_dp, self.root["img_dp"])
        self.meta_dp = os.path.join(cfgs.run.dataset_dp, self.root["meta_dp"])
        self.dsm_txt_fp = os.path.join(cfgs.run.dataset_dp, self.root["dsm_txt_fp"])
        self.dsm_tif_fp = os.path.join(cfgs.run.dataset_dp, self.root["dsm_tif_fp"])
        self.dsm_cls_fp = None
        if self.root.get("dsm_cls_fp") is not None:
            self.dsm_cls_fp = os.path.join(cfgs.run.dataset_dp, self.root["dsm_cls_fp"])
        self.ignore_mask_fp = None
        if self.root.get("ignore_mask_fp") is not None:
            self.ignore_mask_fp = os.path.join(
                cfgs.run.dataset_dp, self.root["ignore_mask_fp"]
            )
        self.zone_string = self.root["zone_string"]
        # load in the geolocation of the dsm scene
        # this is for example used to span up the enu coordinate system
        # used as part of the rpc approximation by VisSat
        self.dsm_center_lons = self.root.get("dsm_center_lons")
        self.dsm_center_lats = self.root.get("dsm_center_lats")
        self.dsm_center_alts = self.root.get("dsm_center_alts", 0.0)
        if split == "train":
            self.data_names = self.root["train_split"]
            if cfgs.run.dataset_limit_train_images:
                self.data_names = self.data_names[: cfgs.run.dataset_limit_train_images]
                logger.info(
                    "Dataset",
                    f"Limit number of training images to run.dataset_limit_train_images={cfgs.run.dataset_limit_train_images}",
                )
        else:
            # we add one image of the train split to the test split for visualization purposes only
            self.data_names = self.root["train_split"][:1] + self.root["test_split"]
        self.data = []
        self.data_unnormalized = None
        self.combined_data = {}
        self.epoch_subsampling_activated = getattr(
            self.cfgs.pipeline, "epoch_subsampling_activated", False
        )

        self.normalization_component = self._init_normalization()
        self.coordinate_system = self._init_coordinate_system()

    def load(self):
        # load data from cache if possible
        # otherwise create from scratch

        if self.has_already_been_cached():
            logger.info("Dataset", "Loading from cache")

        self._init_dataset_creation()

        for idx, name in enumerate(self.data_names):
            show_idx = idx
            if self.split == "test" and idx > 0:
                idx = predefined_val_ts(name)
                if idx is None:
                    logger.info(
                        "Dataset",
                        "Using a test image that is not included in the predefined_val_ts. T index set to zero",
                    )
                    idx = 0
            meta_dict = file_utils.read_dict_from_json(os.path.join(self.meta_dp, name))
            self.data.append(
                self._create_item(
                    name=name,
                    index=idx,
                    meta_dict=meta_dict,
                    load_from_cache=self.has_already_been_cached(),
                )
            )
            logger.info(
                "Dataset",
                f"Image {os.path.basename(name)} loaded ( {show_idx + 1} / {len(self.data_names)} )",
            )

        self.data = self._postprocess_items(self.data)

        self._combine(
            no_subsampling=True
        )  # recombine, if data was changed in post processing

        if self.epoch_subsampling_activated:
            logger.info(
                "Dataset", f"Epoch Subsampling amount: {self._epoch_subsampling_amount()}"
            )
        logger.info("Dataset", f"Finish loading dataset: {self.dataset_name}")

    @abc.abstractmethod
    def _init_dataset_creation(self):
        """
        This method is called once before constructing the data items
        Data that should only be loaded once during dataset creation can be loaded here
        """
        pass

    @abc.abstractmethod
    def _postprocess_items(self, data) -> dict:
        """
        This method is called once after constructing the data items
        This allows for combined changes across all data items
        """
        return data

    def save_to_cache(self):
        if not self.has_already_been_cached():
            for item in self.data:
                self._save_item_to_cache(item)
            logger.info(
                "Dataset", "Saved items to cache for faster loading for future trainings"
            )

    def normalize(self):
        self.data = self.normalization_component.normalize(self.data)
        self._combine()  # need to recombine after normalization

    def initialize_normalization(self, data=None, combined_data=None):
        if combined_data is None:
            self._combine()
            combined_data = self.combined_data
        if data is None:
            data = self.data
        self.normalization_component.initialize(data=data, combined_data=combined_data)

    @abc.abstractmethod
    def _init_normalization(self) -> BaseNormalization:
        """
        Initialize normalization component
        Has to be implemented by the respective dataset
        :return:
        """
        pass

    def force_act_as_test(self):
        """
        This tells the dataset to act as if it's a test dataset,
        independent of the actually loaded split
        --> Each item is a whole training image, instead of singular rays
        """
        self.load_as_if_test = True

    def _combine(self, no_subsampling=False):
        """
        Combines the data items for easier index access if required
        """
        self.combined_data = {}

    def _epoch_subsampling_amount(self):
        cfg_value = self.cfgs.pipeline.epoch_subsampling
        return cfg_value

    def recombine_if_needed(self):
        if self.epoch_subsampling_activated:
            self._combine()

    def __len__(self):
        assert False, "Needs to be implemented by sub classes"

    @abc.abstractmethod
    def __getitem__(self, index) -> T_co:
        assert (
            len(self.data) > 0
        ), "Trying to access data from an instantiated, but unloaded dataset"
        pass

    @abc.abstractmethod
    def _save_item_to_cache(self, data: dict):
        pass

    @abc.abstractmethod
    def has_already_been_cached(self) -> bool:
        return False

    @abc.abstractmethod
    def _create_item(
        self, name: str, index: int, meta_dict: dict, load_from_cache: bool
    ) -> dict:
        pass

    @abc.abstractmethod
    def _init_coordinate_system(self) -> BaseCoordinateSystem:
        pass


class BaseRaysDataset(BaseDataset):
    def __len__(self):
        if self.split == "train" and not self.load_as_if_test:
            if self.epoch_subsampling_activated:
                return len(self.data) * self._epoch_subsampling_amount()
            return self.combined_data["rays"].shape[0]
        else:
            return len(self.data)

    def _epoch_subsampling_amount(self):
        cfg_value = self.cfgs.pipeline.epoch_subsampling
        if 0 <= cfg_value <= 1:
            # if a percentage value is given
            sizes = []
            for item in self.data:
                sizes.append(item["rays"].shape[0])
            return int(cfg_value * max(sizes))
        else:
            return cfg_value

    def _combine(self, no_subsampling=False):
        """
        combines the data items for easier index access
        since for batch loading, each item set = one single ray
        """
        self.combined_data = {}
        for item_index, item in enumerate(self.data):
            # idxs = torch.linspace(
            #     0, item["rays"].shape[0] - 1, steps=item["rays"].shape[0], dtype=torch.int
            # )
            idxs = None
            if self.epoch_subsampling_activated and not no_subsampling:
                # when subsampling activated
                # draw the specified amount of sampled idxs
                idxs = torch.randperm(item["rays"].shape[0] - 1)[
                    : self._epoch_subsampling_amount()
                ]

            for key in item.keys():

                if not torch.is_tensor(item[key]):
                    continue

                if key not in self.combined_data.keys():
                    self.combined_data[key] = item[key]
                else:
                    if idxs is not None:
                        values = item[key][idxs, ...]
                    else:
                        values = item[key]
                    self.combined_data[key] = torch.cat(
                        [self.combined_data[key], values], dim=0
                    )


def predefined_val_ts(img_id):
    """
    Since each image uses the index to access a fitting transient embedding,
    we need to choose a fitting one for the test images
    These values are taken from the original satnerf implementation for the DFC2019 Dataset
    :param img_id: img name without file ending
    :return: the transient embedding id
    """
    img_id = img_id[:-5]  # removes the .json at the end
    aoi_id = img_id[:7]  # area of interest id
    if aoi_id == "JAX_068":
        d = {"JAX_068_013_RGB": 0, "JAX_068_002_RGB": 8, "JAX_068_012_RGB": 1}  # 3
    elif aoi_id == "JAX_004":
        d = {
            "JAX_004_022_RGB": 0,
            "JAX_004_014_RGB": 0,
            "JAX_004_009_RGB": 5,
        }
    elif aoi_id == "JAX_214":
        d = {
            "JAX_214_020_RGB": 0,
            "JAX_214_006_RGB": 8,
            "JAX_214_001_RGB": 18,
            "JAX_214_008_RGB": 2,
        }
    elif aoi_id == "JAX_260":
        d = {"JAX_260_015_RGB": 0, "JAX_260_006_RGB": 3, "JAX_260_004_RGB": 10}
    else:
        return None
    return d.get(img_id, None)
