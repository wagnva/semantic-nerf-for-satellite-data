import cv2
from PIL import Image
import torchvision.transforms as transform
import torch
import numpy as np
from plyflatten import plyflatten
from plyflatten.utils import rasterio_crs, crs_proj
import os
import affine
import rasterio
from framework.visualize import ImageVisualization, BaseVisualization
from framework.datasets import BaseDataset
from framework.util.other import (
    visualize_image,
    scale_image_for_tensorboard,
    SCALE_IMAGE_WIDTH_PIXELS_SMALL,
)
from framework.util.conversions import utm_from_latlon, zonestring_to_hemisphere
from framework.logger import logger


class TensorboardSummaryVisualization(ImageVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        img = results[f"rgb{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = (
            sample["rgbs"].squeeze().view(H, W, 3).permute(2, 0, 1).cpu()
        )  # (3, H, W)
        depth = visualize_image(results[f"depth{typ}"].view(H, W))  # (3, H, W)

        img = scale_image_for_tensorboard(img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL)
        img_gt = scale_image_for_tensorboard(img_gt, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL)
        depth = scale_image_for_tensorboard(depth, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL)

        stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)

        return stack

    def _name(self) -> str:
        return "gt_pred_depth"

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # skip any kind of modification for this vis,
        # results from _visualize are already made for tensorboard
        return img


class AltsVisualization(ImageVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        _, _, alts = dataset.get_latlonalt_from_nerf_prediction(
            sample["rays"].squeeze().cpu(), results[f"depth{typ}"].squeeze().cpu()
        )
        return torch.tensor(alts, device="cpu").view(H, W)  # (H, W)

    def _name(self) -> str:
        return "alts"

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_JET


class FactorVisualization(ImageVisualization):

    def __init__(
        self,
        cfgs,
        send_to_tensorboard: bool,
        save_as_tif: bool,
        factor_name: str,
        viz_name: str = None,
        cmap=cv2.COLORMAP_BONE,
    ) -> None:
        super().__init__(cfgs, send_to_tensorboard, save_as_tif)
        self.factor_name = factor_name
        self.viz_name = viz_name
        if self.viz_name is None:
            self.viz_name = self.factor_name
        self.cmap = cmap

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        if not (f"{self.factor_name}{typ}" in results.keys()):
            logger.error(
                "Visualization",
                f"trying to visualize non-existent factor: {self.factor_name}",
            )
            return None
        results_factor = results[f"{self.factor_name}{typ}"]

        if len(results_factor.shape) == 3:
            if results_factor.shape[2] == 3:
                return (
                    torch.sum(results[f"weights{typ}"].unsqueeze(-1) * results_factor, -2)
                    .view(H, W, 3)
                    .permute(2, 0, 1)
                    .cpu()
                )  # (3, H, W)
            else:
                return torch.sum(
                    results[f"weights{typ}"].unsqueeze(-1)
                    * results[f"{self.factor_name}{typ}"],
                    -2,
                ).view(
                    H, W
                )  # (H, W)
        else:
            if len(results_factor.shape) == 2 and results_factor.shape[1] == 3:
                return results_factor.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            else:
                return results_factor.view(H, W)  # (H, W)

    def _name(self) -> str:
        return self.viz_name

    def _get_visualize_color_scheme(self):
        return self.cmap


class RGBDiffVisualization(ImageVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        img = results[f"rgb{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = (
            sample["rgbs"].squeeze().view(H, W, 3).permute(2, 0, 1).cpu()
        )  # (3, H, W)

        img_diff = img_gt - img

        return torch.abs(img_diff)

    def _name(self) -> str:
        return "RGB_Diff"


class RGBDiffDistanceVisualization(RGBDiffVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        diff = super(RGBDiffDistanceVisualization, self)._visualize(
            pipeline, dataset, sample, results, W, H, typ
        )
        diff = diff.permute(1, 2, 0)  # [H, W, 3]
        diff_squared = torch.square(diff)  # [H, W, 3]
        diff_sum = torch.sum(diff_squared, dim=-1)  # [H, W]
        diff_distance = torch.sqrt(diff_sum)  # [H, W]

        # apply mask as done in the loss

        return diff_distance

    def _name(self) -> str:
        return "RGB_Diff_Distance"

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_BONE
