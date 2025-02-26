import random
import shutil

import fire
import os
import glob
import rasterio
from rasterio.mask import mask
from matplotlib.image import imread
import json
from rpcm.rpc_model import RPCModel
import datetime
import xmltodict
import calendar
from rasterio.windows import Window
import numpy as np
from shapely.geometry import Polygon
from plyflatten.utils import rasterio_crs, crs_proj

from data_prep.processing.step_base import ProcessingStepBase
import data_prep.utils.geo_utils as geo_utils


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)

        # paths from config
        self.aoi_id = cfg.aoi_id  # name of the resulting dataset
        self.location_name = cfg.site.location_name  # name of the input location
        self.location_name_three_letter_string = self.location_name[:3]
        self.truth_dp = step_cfg.get("dfc2019_truth_dp")
        self.metadata_dp = step_cfg.get("dfc2019_metadata_dp")
        self.rgb_dp = step_cfg.get("dfc2019_rgb_dp")
        self.ignore_masks_dp = step_cfg.get("ignore_masks_dp")

        # new paths
        self.output_dp = cfg.output_dp
        self.gt_ifp = os.path.join(self.truth_dp, self.location_name + "_DSM.tif")
        # sorted because for some locations 2 versions of the water mask exist (indicated by postifx _v2)
        self.gt_watermask_ifp = sorted(
            glob.glob(os.path.join(self.truth_dp, f"{self.location_name}_CLS*.tif"))
        )[-1]
        # this one is used for fixing dsm gt georegistration
        self.gt_dsm_txt_fp = os.path.join(self.truth_dp, self.location_name + "_DSM.txt")
        # this one is used for cropping and can be set to an alternative one than the dsm gt one
        self.aoi_txt_ifp = os.path.join(self.truth_dp, self.location_name + "_DSM.txt")
        if cfg.site.alternative_roi_fp:
            self.aoi_txt_ifp = cfg.site.alternative_roi_fp
        self.aoi_txt_fp = os.path.join(self.output_dp, self.location_name + "_DSM.txt")
        self.gt_ofp = os.path.join(self.output_dp, self.location_name + "_DSM.tif")
        self.gt_watermask_ofp = os.path.join(
            self.output_dp, self.location_name + "_CLS.tif"
        )
        self.ignore_mask_ofp = os.path.join(
            self.output_dp, self.location_name + "_ignore_mask.tif"
        )
        self.image_odp = os.path.join(self.output_dp, "rgb_tif")
        self.metas_odp = os.path.join(self.output_dp, "metas")

        # other config vars
        self.max_samples = cfg.files.max_samples
        self.shuffle_dataset = cfg.files.shuffle_dataset
        self.zonestring = cfg.site.zone_string
        self.cfg_alt_min = cfg.site.alt_min
        self.cfg_alt_max = cfg.site.alt_max

        # file filtering
        self.subset_files = cfg.files.subset_files
        self.exclude_files = cfg.files.exclude_files
        self.use_satnerf_test_files = (
            cfg.files.train_test_file_split_method == "use_predefined_test_files"
        )

    def can_be_skipped(self, cfg, state):
        # if we have some .tif in the output dir and some meta files, we probably have run this stage already
        if (
            len(glob.glob(os.path.join(self.image_odp, "*.tif"))) > 0
            and len(glob.glob(os.path.join(self.metas_odp, "*.json"))) > 0
        ):
            return True
        return False

    def run(self, cfg, state):
        os.makedirs(self.image_odp, exist_ok=True)
        os.makedirs(self.metas_odp, exist_ok=True)

        self.create_aoi_txt()

        self.copy_ground_truth()
        # crop gt dsm so that it is from the same region as the other crops
        # self.crop_ground_truth()
        self.fix_ground_truth_crs()
        self.copy_ignore_mask()
        self.convert_tifs()
        self.extract_metas()

    def update_state(self, cfg, state, has_run):
        state["tifs_dp"] = self.image_odp
        state["dsm_fp"] = self.aoi_txt_fp
        state["dsm_tif_fp"] = self.gt_ofp
        state["dsm_cls_fp"] = self.gt_watermask_ofp
        state["metas_dp"] = self.metas_odp
        state["gt_ofp"] = self.gt_ofp
        if self.use_satnerf_test_files:
            state["force_split_test_files"] = self.test_files_satnerf(self.location_name)
        if self.ignore_masks_dp is None or os.path.exists(self.ignore_masks_dp):
            state["ignore_mask_fp"] = self.ignore_mask_ofp

    def copy_ground_truth(self):
        if not os.path.exists(self.gt_ifp):
            print(f"Provided ground truth file does not exist at: {self.gt_ifp}")
        shutil.copy(self.gt_ifp, self.gt_ofp)
        print(f"Copied over ground truth to {self.gt_ofp}")
        shutil.copy(self.gt_watermask_ifp, self.gt_watermask_ofp)
        print(f"Copied over water mask to {self.gt_watermask_ofp}")

    def fix_ground_truth_crs(self):
        # the dfc2019 dsm files have no correct georegistration
        # this applies the information of the provided .txt file to the .tif file
        transform = geo_utils.create_affine_transform_from_aoi_txt(self.gt_dsm_txt_fp)
        # open the original dsm
        with rasterio.open(self.gt_ofp, "r") as src:
            profile = src.profile
            # update the transform
            profile["transform"] = transform
            profile["crs"] = rasterio_crs(crs_proj(self.zonestring, crs_type="UTM"))

            # save to tmp file
            output_tmp_path = self.gt_ofp[:-4] + "_tmp.tif"
            with rasterio.open(output_tmp_path, "w", **profile) as dst:
                # Read the data from the window and write it to the output raster
                dst.write(src.read(1), 1)

            # copy tmp file over and delete it
            os.remove(self.gt_ofp)
            shutil.copy(output_tmp_path, self.gt_ofp)
            os.remove(output_tmp_path)

        # watermask file
        with rasterio.open(self.gt_watermask_ofp, "r") as src:
            profile = src.profile
            # update the transform
            profile["transform"] = transform
            profile["crs"] = rasterio_crs(crs_proj(self.zonestring, crs_type="UTM"))

            # save to tmp file
            output_tmp_path = self.gt_ofp[:-4] + "_tmp.tif"
            with rasterio.open(output_tmp_path, "w", **profile) as dst:
                # Read the data from the window and write it to the output raster
                dst.write(src.read(1), 1)

            # copy tmp file over and delete it
            os.remove(self.gt_watermask_ofp)
            shutil.copy(output_tmp_path, self.gt_watermask_ofp)
            os.remove(output_tmp_path)

    def copy_ignore_mask(self):
        if self.ignore_masks_dp is None or not os.path.isdir(self.ignore_masks_dp):
            # print("No valid ignore mask dp set; Creation skipped")
            return

        from pycocotools.coco import COCO

        coco = COCO(os.path.join(self.ignore_masks_dp, "_annotations.coco.json"))

        for img_id, img in coco.imgs.items():

            if self.location_name not in img["file_name"]:
                continue
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            mask = np.zeros((img["height"], img["width"]))

            for i in range(len(anns)):
                mask[coco.annToMask(anns[i]) == 1] = 1

            with rasterio.open(self.gt_ofp, "r") as ref:
                with rasterio.open(self.ignore_mask_ofp, "w", **ref.profile) as dst:
                    dst.write(mask, 1)

        print("Converted ignore mask into .tif, saving to", self.ignore_mask_ofp)

    def crop_ground_truth(self):
        # copy gt to preserve uncropped
        backup_gt_fp = self.gt_ofp[:-4] + "_uncropped.tif"
        shutil.copy(self.gt_ofp, backup_gt_fp)

        easts, norths = geo_utils.read_aoi_txt(self.gt_dsm_txt_fp, return_utm=True)
        polygon_bbx = Polygon(np.vstack((easts, norths)).T)

        # afterwards crop in based on the aoi information
        with rasterio.open(self.gt_ofp, "r") as src:
            # print("easts", easts)
            # print("norths", norths)
            out_image, out_transform = mask(src, [polygon_bbx], crop=True)

            # Create a new cropped raster to write to
            profile = src.profile
            not_pan = len(out_image.shape) > 2

            height = out_image.shape[0]
            width = out_image.shape[1]
            if not_pan:
                height = out_image.shape[1]
                width = out_image.shape[2]
            profile.update({"height": width, "width": height, "transform": out_transform})

            output_tmp_path = self.gt_ofp[:-4] + "_tmp.tif"
            with rasterio.open(self.gt_ofp, "w", **profile) as dst:
                # Read the data from the window and write it to the output raster
                dst.write(out_image)

            os.remove(self.gt_ofp)
            shutil.copy(output_tmp_path, self.gt_ofp)
            os.remove(output_tmp_path)

            print(f"Cropped gt {os.path.basename(self.gt_ofp)}")

    def extract_metas(self):
        for tif_fp in glob.glob(os.path.join(self.image_odp, "*.tif")):
            basename = os.path.basename(tif_fp)
            if self._should_be_ignored(basename):
                continue

            output = {}

            with rasterio.open(tif_fp, "r") as tif_file:
                # original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")
                # d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])

                output["img"] = os.path.basename(tif_fp)
                output["width"] = int(tif_file.meta["width"])
                output["height"] = int(tif_file.meta["height"])

            # need to find matching imd file
            imd_name = basename[: basename.find("_RGB")][-2:] + ".IMD"
            az, el, capture_time = self.read_imd(
                os.path.join(
                    self.metadata_dp, self.location_name_three_letter_string, imd_name
                )
            )
            output["sun_elevation"] = "+" + str(el)
            output["sun_azimuth"] = str(az)
            output["acquisition_date"] = capture_time.strftime("%Y%m%d%H%M%S")

            if self.cfg_alt_min is not None and self.cfg_alt_max is not None:
                output["min_alt"] = self.cfg_alt_min
                output["max_alt"] = self.cfg_alt_max
                print(
                    f"Use scene altitude boundaries as provided in .cfg: alt_min={self.cfg_alt_min} alt_max={self.cfg_alt_max}"
                )
            else:
                with rasterio.open(self.gt_ofp, "r") as dsm_file:
                    dsm = dsm_file.read()[0, :, :]
                    output["min_alt"] = int(np.round(dsm.min() - 1))
                    output["max_alt"] = int(np.round(dsm.max() + 1))
                    print(
                        f"Use scene altitude boundaries from GT-DSM: alt_min={output['min_alt']} alt_max={output['max_alt']}"
                    )

            # write extracted meta information
            output_fp = os.path.join(
                self.metas_odp, os.path.basename(tif_fp)[:-4] + ".json"
            )

            with open(output_fp, "wt") as ofp:
                json.dump(output, ofp, indent=4)
                print("Extracted meta info", os.path.basename(output_fp))
                ofp.close()

    def read_imd(self, imd_fp):
        with open(imd_fp, "r") as fp:
            lines = fp.readlines()
            for j in range(len(lines)):
                pos = lines[j].find("meanSunAz")
                if pos != -1:
                    last = lines[j].find(";")
                    az = float(lines[j][pos + 11 : last])
                pos = lines[j].find("meanSunEl")
                if pos != -1:
                    last = lines[j].find(";")
                    el = float(lines[j][pos + 11 : last])
                pos = lines[j].find("TLCTime")
                if pos != -1:
                    last = lines[j].find(";")
                    time = datetime.datetime.strptime(
                        lines[j][pos + 11 : last][1:], "%y-%m-%dT%H:%M:%S.%fZ"
                    )

            return az, el, time

    def convert_tifs(self):
        print("Saving tifs into following dir:", self.image_odp)
        image_fps = glob.glob(os.path.join(self.rgb_dp, f"{self.location_name}*_RGB.tif"))

        if self.shuffle_dataset:
            random.shuffle(image_fps)
        else:
            image_fps = list(sorted(image_fps))

        if self.max_samples > 0 and len(image_fps) > self.max_samples:
            image_fps = image_fps[: self.max_samples]

        for image_fp in image_fps:
            if self._should_be_ignored(os.path.basename(image_fp)):
                print("Ignore:", os.path.basename(image_fp))
                continue

            shutil.copy(
                image_fp, os.path.join(self.image_odp, os.path.basename(image_fp))
            )
            # with rasterio.open(image_fp, "r") as src:
            #     profile = src.profile
            #     # update the crs
            #     profile["crs"] = rasterio_crs(crs_proj(self.zonestring, crs_type="UTM"))
            #
            #     with rasterio.open(os.path.join(self.image_odp, os.path.basename(image_fp)), "w", **profile) as dst:
            #         # Read the data from the window and write it to the output raster
            #         dst.write(src.read())
            print(f"Copied {os.path.basename(image_fp)}")

    def _should_be_ignored(self, basename):
        # check if in list of allowed candidates
        if self.subset_files is not None:
            if basename[:-4] not in self.subset_files:
                return True

        # check if in list of exluded files
        if self.exclude_files is not None:
            if basename[:-4] in self.exclude_files:
                return True

        return False

    def create_aoi_txt(self):
        # copy over the aoi_txt
        shutil.copyfile(self.aoi_txt_ifp, self.aoi_txt_fp)

    def test_files_satnerf(self, aoi_name):
        return {
            "JAX_004": ["JAX_004_014_RGB", "JAX_004_009_RGB"],
            "JAX_068": ["JAX_068_002_RGB", "JAX_068_012_RGB"],
            "JAX_214": ["JAX_214_006_RGB", "JAX_214_001_RGB", "JAX_214_008_RGB"],
            "JAX_260": ["JAX_260_006_RGB", "JAX_260_004_RGB"],
        }[aoi_name]
