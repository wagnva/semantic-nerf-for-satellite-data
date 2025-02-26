import json
import rasterio
import os
import glob

from data_prep.processing.step_base import ProcessingStepBase
import data_prep.utils.geo_utils as geo_utils


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)

        # config values
        self.output_dp = os.path.join(cfg.output_dp, "crops")
        self.zone_string = cfg.site.zone_string

        # state read-outs
        self.dsm_fp = state["dsm_fp"]
        self.tifs_dp = state["tifs_dp"]
        self.metas_dp = state["metas_dp"]
        self.gt_fp = state["gt_ofp"]

    def can_be_skipped(self, cfg, state):
        # if we have some .tif in the output dir, we probably have run this stage already
        if len(glob.glob(os.path.join(self.output_dp, "*.tif"))) > 0:
            return True
        return False

    def run(self, cfg, state):
        os.makedirs(self.output_dp, exist_ok=True)

        latlon_box = self.read_lonlat_aoi()

        print("Saving crops into following dir:", self.output_dp)
        for tif_fp in glob.glob(os.path.join(self.tifs_dp, "*.tif")):
            print(f"Cropping {os.path.basename(tif_fp)}")
            output_fp = os.path.join(self.output_dp, os.path.basename(tif_fp))
            width, height = geo_utils.crop_geotiff_lonlat_aoi(
                tif_fp, output_fp, latlon_box
            )
            # width, height = self.crop_geotiff_using_mask(tif_fp, output_fp, latlon_box)
            self.update_meta(tif_fp, width, height)

    def update_state(self, cfg, state, has_run):
        if has_run:
            state["tifs_dp"] = self.output_dp

    def update_meta(self, tif_fp, width, height):
        # find corresponding meta json file and update the width/height value to the cropped values
        meta_fp = os.path.join(self.metas_dp, os.path.basename(tif_fp)[:-4] + ".json")

        with open(meta_fp, "r") as fp:
            dict = json.load(fp)
            dict["width"] = width
            dict["height"] = height
            fp.close()
        with open(meta_fp, "w") as fp:
            json.dump(dict, fp)

    def read_lonlat_aoi(self):
        print(f"Reading roi information from {self.dsm_fp}")
        geo = geo_utils.read_geojson_polygon_from_txt(self.dsm_fp, self.zone_string)
        return geo

    def crop_geotiff_using_mask(self, geotiff_path, output_path, lonlat_aoi):
        # crop in based on the aoi information
        with rasterio.open(geotiff_path, "r") as src:
            profile = src.profile

            out_image, out_transform = rasterio.mask.mask(src, [lonlat_aoi], crop=True)

            # Create a new cropped raster to write to
            not_pan = len(out_image.shape) > 2

            height = out_image.shape[0]
            width = out_image.shape[1]
            if not_pan:
                height = out_image.shape[1]
                width = out_image.shape[2]
            profile.update({"height": width, "width": height, "transform": out_transform})

        with rasterio.open(output_path, "w", **profile) as dst:
            if not_pan:
                dst.write(out_image)
            else:
                dst.write(out_image, 1)
            # dst.update_tags(**tags)
            # dst.update_tags(ns="RPC", **rpc.to_geotiff_dict())

        return width, height
