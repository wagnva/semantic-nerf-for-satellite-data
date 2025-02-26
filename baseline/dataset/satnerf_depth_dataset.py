from torch.utils.data.dataset import T_co
import glob
import os
import numpy as np
import rpcm
import torch

from framework.components.coordinate_systems import BaseCoordinateSystem
from framework.components.normalization import BaseNormalization
import framework.util.file_utils as file_utils
from framework.datasets import BaseRaysDataset
from framework.components.coordinate_systems import (
    CoordinateSystemCustomECEF,
    CoordinateSystemUTM,
)

from baseline.components.normalization import (
    StandardNormalization,
)
from baseline.components.camera_models import construct_rpc_camera_model
from baseline.components.rays import satnerf_construct, construct_sun_dir


class SatNeRFDepthDataset(BaseRaysDataset):
    def __init__(self, cfgs: dict, dataset_name: str, split: str) -> None:
        super().__init__(cfgs, dataset_name, split)
        assert split == "train", "depth dataset is not used for validation"
        assert (
            "points3d_fp" in self.root.keys()
        ), "trying to create depth dataset on not bundle adjusted input dataset"
        self.points3d_fp = os.path.join(cfgs.run.dataset_dp, self.root["points3d_fp"])
        self.kp_weights = None
        self.tie_points = None
        # for the depth dataset, the ray subsampling is always deactivated
        self.ray_subsampling_activated = False

    def _init_dataset_creation(self):
        """
        This method is called once before constructing the data items
        Data that should only be loaded once during dataset creation can be loaded here
        """
        self.tie_points = np.load(self.points3d_fp)
        self.kp_weights = self.load_keypoint_weights_for_depth_supervision(
            self.data_names, self.tie_points
        )

    def _create_item(
        self, name: str, index: int, meta_dict: dict, load_from_cache: bool
    ) -> dict:
        d = meta_dict
        img_id = file_utils.get_file_id(d["img"])

        assert "keypoints" in d.keys(), f"No 'keypoints' field was found in {name}"

        pts2d = np.array(d["keypoints"]["2d_coordinates"])
        pts3d = np.array(self.tie_points[d["keypoints"]["pts3d_indices"], :])

        if type(self.coordinate_system) is not CoordinateSystemCustomECEF:
            # pts3d is in ecef coordinates
            # convert to lat,lon,alt and then to the used coordinate system
            lat, lon, alt = CoordinateSystemCustomECEF(self).to_lat_lon(
                pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
            )
            eastings, northings, alts = self.coordinate_system.from_latlon(lat, lon, alt)
            pts3d = np.stack([eastings, northings, alts], axis=1)

        camera_model = construct_rpc_camera_model(d)

        # build the sparse batch of rays for depth supervision
        cols, rows = pts2d.T
        min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
        rays = satnerf_construct(
            camera_model,
            self.coordinate_system,
            rows=rows,
            cols=cols,
            min_alt=min_alt,
            max_alt=max_alt,
        )

        n_rays = rays.shape[0]

        pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)

        # normalize both the rays and pts3d
        rays = self.normalization_component.normalize_single({"rays": rays})["rays"]
        pts3d = self.normalization_component.normalize_xyz(pts3d)

        depths = torch.linalg.norm(pts3d - rays[:, :3], axis=1)
        weights = torch.from_numpy(self.kp_weights[d["keypoints"]["pts3d_indices"]]).type(
            torch.FloatTensor
        )

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
            "depths": depths[:, np.newaxis],
            "weights": weights[:, np.newaxis],
            "extras": extras,
            "name": img_id,
            "w": int(d["width"]),
            "h": int(d["height"]),
        }

        return data_item

    def __getitem__(self, index) -> T_co:
        if self.split == "train" and not self.load_as_if_test:
            return {
                "rays": self.combined_data["rays"][index],
                "depths": self.combined_data["depths"][index],
                "weights": self.combined_data["weights"][index],
                "extras": self.combined_data["extras"][index],
                # "name": self.combined_data["name"],
                # "w": self.combined_data["w"],
                # "h": self.combined_data["h"],
            }
        else:
            # if test split, return whole images at once
            d = self.data[index]
            # useful metadata about test image
            d["split"] = "train" if index == 0 else "test"
            d["img_fp"] = os.path.join(self.img_dp, d["name"] + ".tif")
            return d

    def load_keypoint_weights_for_depth_supervision(self, data_names: list, tie_points):
        n_pts = tie_points.shape[0]
        n_cams = len(data_names)
        reprojection_errors = np.zeros((n_pts, n_cams), dtype=np.float32)

        # results are saved in ecef coordinate system, no matter the coordinate system used for training
        coordinate_system = CoordinateSystemCustomECEF(self)
        for t, name in enumerate(data_names):
            d = file_utils.read_dict_from_json(os.path.join(self.meta_dp, name))

            if "keypoints" not in d.keys():
                raise ValueError("No 'keypoints' field was found in {}".format(name))

            pts2d = np.array(d["keypoints"]["2d_coordinates"])
            pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])

            camera_model = construct_rpc_camera_model(d)

            lat, lon, alt = coordinate_system.to_lat_lon(
                pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
            )
            col, row = camera_model.projection(lon, lat, alt)
            pts2d_reprojected = np.vstack((col, row)).T
            errs_obs_current_cam = np.linalg.norm(pts2d - pts2d_reprojected, axis=1)
            reprojection_errors[d["keypoints"]["pts3d_indices"], t] = errs_obs_current_cam

        e = np.sum(reprojection_errors, axis=1)
        e_mean = np.mean(e)
        weights = np.exp(-((e / e_mean) ** 2))

        return weights

    def _init_coordinate_system(self) -> BaseCoordinateSystem:
        if self.cfgs.pipeline.use_utm_coordinate_system:
            return CoordinateSystemUTM(self)
        return CoordinateSystemCustomECEF(self)

    def _save_item_to_cache(self, data: dict):
        # this dataset doesn't cache anything
        pass

    def has_already_been_cached(self) -> bool:
        # this dataset doesn't cache anything itself
        return False

    def _init_normalization(self) -> BaseNormalization:
        cache_name = "normalization"
        if self.cfgs.pipeline.use_utm_coordinate_system:
            cache_name = "normalization_utm"

        return StandardNormalization(self.cfgs, cache_name=cache_name)

    def normalize(self):
        # no actual normalization here,
        # since the data is already normalized during dataset creation
        # different in comparison to rgb datasets
        pass
