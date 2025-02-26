import json

import torch
import os
import numpy as np
import glob
from torch.utils.data.dataset import T_co
import rpcm

from framework.components.coordinate_systems import BaseCoordinateSystem
from framework.components.normalization import BaseNormalization
from framework.datasets import BaseRaysDataset
import framework.util.img_utils as img_utils
import framework.util.file_utils as file_utils
from framework.components.rays import load_from_disk, save_to_disk
from framework.logger import logger
import framework.util.sat_utils as sat_utils
from framework.components.coordinate_systems import (
    CoordinateSystemCustomECEF,
    CoordinateSystemUTM,
)

from baseline.components.camera_models import CameraModelRPC
from baseline.components.normalization import (
    StandardNormalization,
)
from baseline.components.rays import satnerf_construct, construct_sun_dir
from baseline.components.camera_models import construct_rpc_camera_model


class SatNeRFDataset(BaseRaysDataset):
    def __init__(self, cfgs, dataset_name: str, split: str) -> None:
        self.cache_name = "rays"
        self.norm_cache_name = "normalization"
        if cfgs.pipeline.use_utm_coordinate_system:
            self.cache_name = "rays_utm"
            self.norm_cache_name = "normalization_utm"
        super().__init__(cfgs, dataset_name, split)

    def has_already_been_cached(self) -> bool:
        if not self.cache.exists(self.cache_name):
            return False
        for name in self.data_names:
            if (
                len(
                    glob.glob(
                        os.path.join(
                            self.cache.dir_path(self.cache_name), f"{name[:-5]}.data"
                        )
                    )
                )
                == 0
            ):
                return False
        return True

    def _create_item(
        self, name: str, index: int, meta_dict: dict, load_from_cache: bool
    ) -> dict:
        """
        Instantiate ray and load rgb
        """
        d = meta_dict
        img_fp = os.path.join(self.img_dp, d["img"])
        img_id = file_utils.get_file_id(d["img"])

        # get rgb colors
        rgbs = img_utils.load_tensor_from_rgb_geotiff(img_fp)

        h, w = int(d["height"]), int(d["width"])
        min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])

        camera_model = self.construct_camera_model(d)

        if load_from_cache:
            rays = load_from_disk(img_id, self.cache.dir_path(self.cache_name))
        else:
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = satnerf_construct(
                camera_model,
                self.coordinate_system,
                rows=rows,
                cols=cols,
                min_alt=min_alt,
                max_alt=max_alt,
            )

        # check if dimensions match
        assert (
            rgbs.shape[0] == rays.shape[0]
        ), f"rgb & rays dimensions dont match in: {os.path.basename(name)}"

        n_rays = rays.shape[0]

        sun_dirs = construct_sun_dir(
            float(d["sun_elevation"]),
            float(d["sun_azimuth"]),
            n_rays,
        )

        # timestamps, used later on for embeddings of transient features based on viewing angle
        ts = index * torch.ones(n_rays, 1)

        extras = torch.hstack([sun_dirs, ts])

        data_item = {
            "rays": rays,
            "rgbs": rgbs,
            "extras": extras,
            "name": img_id,
            "w": w,
            "h": h,
            "alt_min": min_alt,
            "alt_max": max_alt,
        }

        return data_item

    def construct_camera_model(self, d):
        return construct_rpc_camera_model(d)

    def __getitem__(self, index) -> T_co:
        if self.split == "train" and not self.load_as_if_test:
            return {
                "rays": self.combined_data["rays"][index],
                "rgbs": self.combined_data["rgbs"][index],
                "extras": self.combined_data["extras"][index],
                # "name": self.combined_data["name"],
                # "w": self.combined_data["w"],
                # "h": self.combined_data["h"],
                # "alt_min": self.combined_data["alt_min"],
                # "alt_max": self.combined_data["alt_max"],
            }
        else:
            # if test split, return whole images at once
            d = self.data[index]
            # useful metadata about test image
            d["split"] = "train" if index == 0 else "test"
            d["img_fp"] = os.path.join(self.img_dp, d["name"] + ".tif")
            return d

    def _save_item_to_cache(self, data: dict):
        save_to_disk(data["rays"], data["name"], self.cache.dir_path(self.cache_name))

    def _init_coordinate_system(self) -> BaseCoordinateSystem:
        if self.cfgs.pipeline.use_utm_coordinate_system:
            return CoordinateSystemUTM(self)
        return CoordinateSystemCustomECEF(self)

    def _init_dataset_creation(self):
        pass

    def _init_normalization(self) -> BaseNormalization:
        return StandardNormalization(self.cfgs, cache_name=self.norm_cache_name)

    def get_xyz_from_nerf_prediction(self, rays, depth):
        """
        Computes the xyz coordinates for rays with a given end depth
        :param rays: the rays
        :param depth: the end points of the rays at estimated depth
        :return: normalized xyz endpoints of rays
        """
        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        rays = rays.double()
        depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)

        return xyz_n

    def get_latlonalt_from_nerf_prediction(self, rays, depth):
        """
        Convert NeRF rays with a specified depth into Lat/Lon/Alt 3D Points
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        xyz_n = self.get_xyz_from_nerf_prediction(rays, depth)
        lats, lons, alts = self.get_latlonalt_from_points(xyz_n)
        return lats, lons, alts

    def get_latlonalt_from_points(self, points):
        """
        Convert normalized points into Lat/Lon/Alt 3D Points
        Args:
            points: (h*w, 3) tensor of input, normalized points
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """
        # denormalize prediction to obtain world coordinates
        xyz = self.normalization_component.denormalize({"xyz": points})
        # convert to lat-lon-alt
        xyz = xyz.data.numpy()
        lats, lons, alts = self.coordinate_system.to_lat_lon(
            xyz[:, 0], xyz[:, 1], xyz[:, 2]
        )
        return lats, lons, alts
