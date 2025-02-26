import json
import shutil
import random
import rasterio
import rpcm
import numpy as np
import os
import glob
import sys
from bundle_adjust import geo_utils, loader
from bundle_adjust.cam_utils import SatelliteImage
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline

from data_prep.processing.step_base import ProcessingStepBase
import data_prep.utils.geo_utils as own_geo_utils


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)

        self.output_fp = os.path.join(cfg.output_dp, "root.json")

        # config values
        self.location_name = cfg.site.location_name
        ba_step_cfg = list(filter(lambda x: "bundle_adjustment" in x.file, cfg.steps))
        self.ba_enabled = len(ba_step_cfg) > 0 and ba_step_cfg[0].enabled
        self.zone_string = cfg.site.zone_string
        self.shuffle_dataset = (
            cfg.files.shuffle_dataset
            or cfg.files.train_test_file_split_method == "random_test_files"
        )
        if "use_custom_test_files" == cfg.files.train_test_file_split_method:
            self.force_split_test_files = cfg.files.custom_test_files
        else:
            self.force_split_test_files = None
        self.force_amount_test_files = None
        if "use_fixed_test_file_amount" == cfg.files.train_test_file_split_method:
            self.force_amount_test_files = cfg.files.fixed_test_file_amount

        # state read-outs
        self.tifs_dp = state["tifs_dp"]
        self.metas_dp = state["metas_dp"]
        self.ba_files_dp = state["ba_files_dp"]
        self.dsm_txt_fp = state["dsm_fp"]
        self.dsm_tif_fp = state["dsm_tif_fp"]
        self.dsm_cls_fp = state.get("dsm_cls_fp")
        self.ignore_mask_fp = state.get("ignore_mask_fp")
        self.state_force_split_test_files = state.get("force_split_test_files", None)

    def can_be_skipped(self, cfg, state):
        # always run
        return False

    def run(self, cfg, state):
        """
        This step creates a root.json file containing all the information about this dataset
        This includes paths to the metas, images, ba files and training/test split
        """

        output = {
            "aoi_name": self.location_name,
            "meta_dp": os.path.relpath(self.metas_dp, cfg.output_dp),
            "img_dp": os.path.relpath(self.tifs_dp, cfg.output_dp),
            "dsm_txt_fp": os.path.relpath(self.dsm_txt_fp, cfg.output_dp),
            "dsm_tif_fp": os.path.relpath(self.dsm_tif_fp, cfg.output_dp),
            "zone_string": self.zone_string,
        }

        if self.dsm_cls_fp is not None:
            output["dsm_cls_fp"] = os.path.relpath(self.dsm_cls_fp, cfg.output_dp)

        if self.ignore_mask_fp is not None:
            output["ignore_mask_fp"] = os.path.relpath(self.ignore_mask_fp, cfg.output_dp)

        if self.ba_enabled:
            output["points3d_fp"] = os.path.join(
                self.ba_files_dp, "ba_params", "pts3d.npy"
            )
            output["points3d_fp"] = os.path.relpath(output["points3d_fp"], cfg.output_dp)

        # create the train/test split
        json_files = [
            os.path.basename(p) for p in glob.glob(os.path.join(self.metas_dp, "*.json"))
        ]
        if self.shuffle_dataset:
            random.shuffle(json_files)
        else:
            json_files = list(sorted(json_files))

        if self.force_amount_test_files is not None:
            train_samples = json_files[self.force_amount_test_files :]
            test_samples = json_files[: self.force_amount_test_files]
        elif self.force_split_test_files is not None:
            test_samples = list(map(lambda x: f"{x}.json", self.force_split_test_files))
            train_samples = list(filter(lambda x: x not in test_samples, json_files))
        elif self.state_force_split_test_files is not None:
            test_samples = list(
                map(lambda x: f"{x}.json", self.state_force_split_test_files)
            )
            train_samples = list(filter(lambda x: x not in test_samples, json_files))
        else:
            train_samples, test_samples = self.create_train_test_splits(json_files)

        # since depending on the configuration, the name of the test items are taken directly from config
        # make sure they are correct item names that are actually present in the dataset
        for test_json_name in test_samples:
            assert os.path.exists(
                os.path.join(self.metas_dp, test_json_name)
            ), "One of the specified test files can not be found. Make sure you used the correct name (without .json)"

        output["train_split"] = train_samples
        output["test_split"] = test_samples

        # load in the geolocation of the dsm scene
        # this is for example used to span up the enu coordinate system
        # used as part of the rpc approximation by VisSat
        lonlat_bbx = own_geo_utils.read_geojson_polygon_from_txt(
            self.dsm_txt_fp, self.zone_string
        )
        output["dsm_center_lons"] = lonlat_bbx["center"][0]
        output["dsm_center_lats"] = lonlat_bbx["center"][1]

        with open(self.output_fp, "wt+") as fp:
            json.dump(output, fp, indent=4)

        print("Finished creating file at:", self.output_fp)

    def update_state(self, cfg, state, has_run):
        pass

    def create_train_test_splits(
        self,
        input_sample_ids,
        test_percent=0.15,
        min_test_samples=2,
        max_samples=-1,
    ):
        def shuffle_array(array):
            import random

            v = array.copy()
            random.shuffle(v)
            return v

        n_samples = len(input_sample_ids)
        input_sample_ids = np.array(input_sample_ids)
        all_indices = shuffle_array(np.arange(n_samples))

        if max_samples >= 0 and max_samples < n_samples:
            n_samples = max_samples
            all_indices = all_indices[:max_samples]

        n_test = max(min_test_samples, int(test_percent * n_samples))
        n_train = n_samples - n_test

        train_indices = all_indices[:n_train]
        test_indices = all_indices[-n_test:]

        train_samples = input_sample_ids[train_indices].tolist()
        test_samples = input_sample_ids[test_indices].tolist()

        return train_samples, test_samples
