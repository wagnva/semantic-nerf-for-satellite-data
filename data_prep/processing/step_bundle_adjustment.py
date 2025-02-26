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


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)

        # config values
        self.output_dp = os.path.join(cfg.output_dp, "ba_files")
        self.tifs_ba_odp = os.path.join(cfg.output_dp, "rgb_tif_ba")

        # state read-outs
        self.tifs_dp = state["tifs_dp"]

    def can_be_skipped(self, cfg, state):
        # check if one of the last files was saved -> ba should have run through correctly once before
        if os.path.exists(os.path.join(self.output_dp, "ba_params", "pts2d.npy")):
            return True
        return False

    def run(self, cfg, state):
        myimages = sorted(glob.glob(self.tifs_dp + "/*.tif"))
        myrpcs = [rpcm.rpc_from_geotiff(p) for p in myimages]
        input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
        ba_input_data = {}
        ba_input_data["in_dir"] = self.tifs_dp
        ba_input_data["out_dir"] = self.output_dp
        ba_input_data["images"] = input_images
        print("Input data set!\n")

        # redirect all prints to a bundle adjustment logfile inside the output directory
        os.makedirs(ba_input_data["out_dir"], exist_ok=True)
        path_to_log_file = "{}/bundle_adjust.log".format(ba_input_data["out_dir"])
        print("Running bundle adjustment for RPC model refinement ...")
        print("Path to log file: {}".format(path_to_log_file))
        log_file = open(path_to_log_file, "w+")
        sys.stdout = log_file
        sys.stderr = log_file
        # run bundle adjustment
        # tracks_config = {'FT_reset': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_K": 300}
        tracks_config = {
            "FT_reset": False,
            "FT_save": True,
            "FT_sift_detection": "s2p",
            "FT_sift_matching": "epipolar_based",
        }
        ba_extra = {"cam_model": "rpc"}
        ba_pipeline = BundleAdjustmentPipeline(
            ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra
        )
        ba_pipeline.run()
        # close logfile
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__
        log_file.close()
        print("... done !")
        print("Path to output files: {}".format(ba_input_data["out_dir"]))

        # save all bundle adjustment parameters in a temporary directory
        ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
        os.makedirs(ba_params_dir, exist_ok=True)
        np.save(
            os.path.join(ba_params_dir, "pts_ind.npy"),
            ba_pipeline.ba_params.pts_ind,
        )
        np.save(
            os.path.join(ba_params_dir, "cam_ind.npy"),
            ba_pipeline.ba_params.cam_ind,
        )
        np.save(
            os.path.join(ba_params_dir, "pts3d.npy"),
            ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform,
        )
        np.save(
            os.path.join(ba_params_dir, "pts2d.npy"),
            ba_pipeline.ba_params.pts2d,
        )
        frames_in_use = [
            ba_pipeline.images[idx].geotiff_path
            for idx in ba_pipeline.ba_params.cam_prev_indices
        ]
        loader.save_list_of_paths(
            os.path.join(ba_params_dir, "geotiff_paths.txt"), frames_in_use
        )

        self.update_rpc_model_in_tif(os.path.join(ba_pipeline.out_dir, "rpcs_adj"))

    def update_rpc_model_in_tif(self, rpc_adj_dp):
        os.makedirs(self.tifs_ba_odp, exist_ok=True)
        for tif_fp in glob.glob(os.path.join(self.tifs_dp, "*.tif")):
            with rasterio.open(tif_fp, "r") as src:
                rpc_fp = os.path.join(
                    rpc_adj_dp, os.path.basename(tif_fp)[:-4] + ".rpc_adj"
                )
                rpc = rpcm.rpc_from_rpc_file(rpc_fp)

                tif_ba_ofp = os.path.join(self.tifs_ba_odp, os.path.basename(tif_fp))
                with rasterio.open(tif_ba_ofp, "w", **src.profile) as dst:
                    dst.write(src.read())
                    dst.update_tags(ns="RPC", **rpc.to_geotiff_dict())

        print("Wrote updated ba rpc files to .tif")

    def update_state(self, cfg, state, has_run):
        state["ba_files_dp"] = self.output_dp
