import json
import os
import numpy as np
import torch

from framework.components.normalization import BaseNormalization
from framework.components.rays import ray_component_fn
from framework.logger import logger


class StandardNormalization(BaseNormalization):
    def __init__(self, cfgs, cache_name="normalization") -> None:
        super().__init__(cfgs)
        self.cache_name = cache_name
        self.cache_fp = os.path.join(
            self.cache.dir_path(self.cache_name), "norm_params.json"
        )

    def normalize_single(self, data: dict) -> dict:
        center, range = self.calculate_center_range()

        ray = data["rays"]
        origins = ray_component_fn(ray, "origins")
        origins = self.normalize_xyz(origins)
        ray_component_fn(ray, "origins", origins)

        nears = ray_component_fn(ray, "near")
        nears /= range
        ray_component_fn(ray, "near", nears)

        fars = ray_component_fn(ray, "far")
        fars /= range
        ray_component_fn(ray, "far", fars)

        data["rays"] = ray
        return data

    def normalize_xyz(self, xyz: torch.tensor):
        center, range = self.calculate_center_range()
        # subtract offsets
        xyz[:, 0] -= center[0]
        xyz[:, 1] -= center[1]
        xyz[:, 2] -= center[2]
        # scale by range
        xyz[:, 0] /= range
        xyz[:, 1] /= range
        xyz[:, 2] /= range
        return xyz

    def denormalize(self, item: dict) -> torch.tensor:
        assert "xyz" in item.keys(), "only xyz denormalization implemented"
        xyz = item["xyz"]
        center, range = self.calculate_center_range()
        xyz = xyz * range
        xyz[:, 0] += center[0]
        xyz[:, 1] += center[1]
        xyz[:, 2] += center[2]
        return xyz

    def calculate_center_range(self):
        assert {
            "X_offset",
            "Y_offset",
            "Z_offset",
            "X_scale",
            "Y_scale",
            "Z_scale",
        }.issubset(
            self.norm_params
        ), "Normalization parameters don't have the required values. Something went wrong with initialization"

        d = self.norm_params
        center = torch.tensor(
            [float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])]
        )
        range = torch.max(
            torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])])
        )
        return center, range

    def _calculate_normalization_params(self, data: list, combined_data: dict) -> dict:
        combined_rays = combined_data["rays"]
        all_origins = ray_component_fn(combined_rays, "origins")
        all_dirs = ray_component_fn(combined_rays, "directions")
        all_fars = ray_component_fn(combined_rays, "fars")
        near_points = all_origins
        far_points = all_origins + all_fars * all_dirs
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = self.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = self.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = self.rpc_scaling_params(all_points[:, 2])

        return d

    def rpc_scaling_params(self, v):
        """
        find the scale and offset of a vector
        """
        vec = np.array(v).ravel()
        scale = (vec.max() - vec.min()) / 2
        offset = vec.min() + scale
        return scale, offset
