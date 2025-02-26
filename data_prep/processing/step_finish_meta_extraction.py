import json
import shutil
import rasterio
import rpcm
import numpy as np
import os
import glob
import srtm4
from bundle_adjust import geo_utils, loader

from data_prep.processing.step_base import ProcessingStepBase


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)

        # config values
        ba_step_cfg = list(filter(lambda x: "bundle_adjustment" in x.file, cfg.steps))
        self.ba_enabled = len(ba_step_cfg) > 0 and ba_step_cfg[0].enabled
        self.output_dp = cfg.output_dp

        # state read-outs
        self.tifs_dp = state["tifs_dp"]
        self.metas_dp = state["metas_dp"]
        self.ba_files_dp = state["ba_files_dp"]

    def can_be_skipped(self, cfg, state):
        # if we have some .tif in the output dir, and one of them has one of the attr we set in this stage already,
        # we have probably run this stage already
        meta_fps = glob.glob(os.path.join(self.metas_dp, "*.json"))
        if len(meta_fps) > 0:
            with open(meta_fps[0], "r") as fp:
                d = json.load(fp)
                if "geojson" in d:
                    return True
        return False

    def run(self, cfg, state):
        for meta_fp in glob.glob(os.path.join(self.metas_dp, "*.json")):
            with open(meta_fp, "r") as fp:
                d = json.load(fp)

            tif_fp = os.path.join(self.tifs_dp, os.path.basename(meta_fp)[:-5] + ".tif")
            with rasterio.open(os.path.join(tif_fp)) as src:
                # load the rpc model of the tif
                original_rpc = rpcm.RPCModel(src.tags(ns="RPC"), dict_format="geotiff")

            # set the geojson of the aoi
            # this is done using the original rpc, even if using ba (based on original script)
            d["geojson"] = self.get_image_lonlat_aoi(
                original_rpc, d["width"], d["height"]
            )

            # if bundle adjustment has run, we need to use the rpc info with the ba adjusted one
            if self.ba_enabled:
                rpc_path = os.path.join(
                    self.ba_files_dp,
                    "rpcs_adj",
                    f"{os.path.basename(meta_fp)[:-5]}.rpc_adj",
                )
                d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__

                # additional fields for depth supervision
                geotiff_paths = loader.load_list_of_paths(
                    os.path.join(self.ba_files_dp, "ba_params", "geotiff_paths.txt")
                )
                geotiff_paths = [
                    p.replace("/pan_crops/", "/crops/") for p in geotiff_paths
                ]
                geotiff_paths = [p.replace("PAN.tif", "RGB.tif") for p in geotiff_paths]
                ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
                ba_kps_pts3d_ind = np.load(
                    os.path.join(self.ba_files_dp, "ba_params", "pts_ind.npy")
                )
                ba_kps_cam_ind = np.load(
                    os.path.join(self.ba_files_dp, "ba_params", "cam_ind.npy")
                )
                ba_kps_pts2d = np.load(
                    os.path.join(self.ba_files_dp, "ba_params", "pts2d.npy")
                )

                cam_idx = ba_geotiff_basenames.index(d["img"])
                d["keypoints"] = {
                    "2d_coordinates": ba_kps_pts2d[ba_kps_cam_ind == cam_idx, :].tolist(),
                    "pts3d_indices": ba_kps_pts3d_ind[ba_kps_cam_ind == cam_idx].tolist(),
                }

            else:
                d["rpc"] = original_rpc.__dict__

            # write the updated meta information
            with open(meta_fp, "w") as fp:
                json.dump(d, fp, indent=4)
                print(
                    f"Finished extracting meta information for {os.path.basename(meta_fp)}"
                )

    def update_state(self, cfg, state, has_run):
        pass

    def get_image_lonlat_aoi(self, rpc, h, w):
        z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
        cols, rows, alts = [0, w, w, 0], [0, 0, h, h], [z] * 4
        lons, lats = rpc.localization(cols, rows, alts)
        lonlat_coords = np.vstack((lons, lats)).T
        geojson_polygon = {
            "coordinates": [lonlat_coords.tolist()],
            "type": "Polygon",
        }
        x_c = lons.min() + (lons.max() - lons.min()) / 2
        y_c = lats.min() + (lats.max() - lats.min()) / 2
        geojson_polygon["center"] = [x_c, y_c]
        return geojson_polygon
