import os
import torch

from baseline.dataset.satnerf_dataset import SatNeRFDataset
from framework.util.img_utils import load_tensor_from_cls_geotiff


class SemanticDataset(SatNeRFDataset):

    def __init__(self, cfgs, dataset_name: str, split: str) -> None:
        super().__init__(cfgs, dataset_name, split)
        self.semantic_dataset_name = f"semantic_dp_{cfgs.pipeline.semantic_dataset_type}"
        assert set([self.semantic_dataset_name, "semantic_cls_labels"]) <= set(
            self.root.keys()
        ), "trying to use a semantic pipeline on a dataset not containing semantic data"
        self.semantic_dp = os.path.join(
            cfgs.run.dataset_dp, self.root[self.semantic_dataset_name]
        )
        self.labels_are_corrupted = "corrupted" in cfgs.pipeline.semantic_dataset_type
        if self.labels_are_corrupted:
            # path to the not corrupted dataset
            self.semantic_non_corrupted_dp = os.path.join(
                cfgs.run.dataset_dp,
                self.root["semantic_dp_" + cfgs.pipeline.semantic_dataset_type[:-10]],
            )

        self.semantic_no_cars_dp = None
        if self.root.get(self.semantic_dataset_name + "_no_cars"):
            # path to the dataset w/o cars
            self.semantic_no_cars_dp = os.path.join(
                cfgs.run.dataset_dp,
                self.root[self.semantic_dataset_name + "_no_cars"],
            )
        self.semantic_cls_labels = self.root["semantic_cls_labels"]
        self.semantic_n_classes = len(self.semantic_cls_labels.keys())
        self.car_cls_idx = None
        if "cars" in self.semantic_cls_labels.values():
            self.car_cls_idx = filter(
                lambda x: x[1] == "cars", self.semantic_cls_labels.items()
            )
            self.car_cls_idx = int(next(self.car_cls_idx)[0])

        self.sparsity_n_images = self.cfgs.pipeline.sparsity_n_images

    def _create_item(
        self, name: str, index: int, meta_dict: dict, load_from_cache: bool
    ) -> dict:
        # create rays as normal
        item = super()._create_item(name, index, meta_dict, load_from_cache)

        img_fp = os.path.join(self.semantic_dp, meta_dict["img"][:-7] + "CLS.tif")
        labels = load_tensor_from_cls_geotiff(img_fp)

        # if sparsity is set (not -1): set ignore mask if index above n sparsity images
        # only if split==train
        sparsity_mask = torch.ones(
            labels.shape[0], device=labels.device, dtype=torch.bool
        )  # (N_rays)
        if self.split == "train" and 0 < self.sparsity_n_images <= index:
            sparsity_mask = torch.zeros(
                labels.shape[0], device=labels.device, dtype=torch.bool
            )  # (N_rays)

        item["semantic"] = labels
        item["semantic_sparsity_mask"] = sparsity_mask

        if self.labels_are_corrupted:
            img_fp = os.path.join(
                self.semantic_non_corrupted_dp, meta_dict["img"][:-7] + "CLS.tif"
            )
            labels = load_tensor_from_cls_geotiff(img_fp)
            item["semantic_non_corrupted"] = labels

        if self.semantic_no_cars_dp:
            img_fp = os.path.join(
                self.semantic_no_cars_dp, meta_dict["img"][:-7] + "CLS.tif"
            )
            labels = load_tensor_from_cls_geotiff(img_fp)
            item["semantic_no_cars"] = labels

        return item

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.split == "train" and not self.load_as_if_test:
            item["semantic"] = self.combined_data["semantic"][index]
            item["semantic_sparsity_mask"] = self.combined_data["semantic_sparsity_mask"][
                index
            ]
        return item
