import cv2
from PIL import Image
import torchvision.transforms as transform
import torch
import glob
import numpy as np
from plyflatten import plyflatten
from plyflatten.utils import rasterio_crs, crs_proj
import os
import affine
import rasterio
import datetime
import eval.utils.dsmr as dsmr

from framework.util.conversions import utm_from_latlon, zonestring_to_hemisphere


def get_utm_cloud(
    lats: np.ndarray, lons: np.ndarray, alts: np.ndarray
) -> tuple[np.ndarray, str]:
    """
    Get an utm cloud from lats, lons, alts
    :param lats: latitudes of points
    :param lons: longitudes of points
    :param alts: altitude of points
    :return: numpy stacked utm cloud, zone_string of utm zone
    """
    easts, norths, zone_string = utm_from_latlon(lats, lons)
    cloud = np.vstack([easts, norths, alts]).T
    return cloud, zone_string


def create_dsm_cloud_from_nerf(dataset, rays, depths):
    lats, lons, alts = dataset.get_latlonalt_from_nerf_prediction(rays, depths)
    cloud, zone_string = get_utm_cloud(lats, lons, alts)
    return cloud


def create_dsm(
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    dsm_path=None,
    roi_txt=None,
):
    """
    Compute a DSM from given lats, lons and alts
    :param lats: latitudes of points
    :param lons: longitudes of points
    :param alts: altitude of points
    :param dsm_path: where to save the dsm to. If not given, the dsm is just returned
    :param roi_txt: compute dsm only in the given bounds of the region of interest
    :return: the dsm as tif
    """
    cloud, zone_string = get_utm_cloud(lats, lons, alts)

    # (optional) read region of interest, where lidar GT is available
    if roi_txt is not None:
        gt_roi_metadata = np.loadtxt(roi_txt)
        xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
        xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        resolution = gt_roi_metadata[3]
        yoff += ysize * resolution  # weird but seems necessary ?
        print("using roi_txt with resolution:", resolution)
    else:
        resolution = 0.5
        xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
        ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
        xoff = np.floor(xmin / resolution) * resolution
        xsize = int(1 + np.floor((xmax - xoff) / resolution))
        yoff = np.ceil(ymax / resolution) * resolution
        ysize = int(1 - np.floor((ymin - yoff) / resolution))

    # run plyflatten
    dsm = plyflatten(
        cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf")
    )
    crs = rasterio_crs(crs_proj(zonestring_to_hemisphere(zone_string), crs_type="UTM"))

    #        ul_e, ul_n, size, resolution = bbox
    #    ul_n += size * resolution  # necessary to fix rotation
    #         transform = rasterio.Affine(resolution, 0.0, ul_e, 0.0, -resolution, ul_n)

    # (optional) write dsm to disk
    if dsm_path is not None:
        os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
        profile = {}
        profile["dtype"] = dsm.dtype
        profile["height"] = dsm.shape[0]
        profile["width"] = dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        profile["crs"] = crs
        profile["transform"] = affine.Affine(
            resolution, 0.0, xoff, 0.0, -resolution, yoff
        )

        with rasterio.open(dsm_path, "w", **profile) as f:
            f.write(dsm[:, :, 0], 1)

    return dsm


def compute_dsm_and_mae(dataset, rays, depths, output_dp, img_name, epoch):
    lats, lons, alts = dataset.get_latlonalt_from_nerf_prediction(rays, depths)
    return compute_dsm_and_mae_from_latlon(
        lats, lons, alts, dataset, output_dp, img_name, epoch
    )


def compute_dsm_and_mae_from_latlon(
    lats, lons, alts, dataset, output_dp, img_name, epoch
):
    output_fp = os.path.join(output_dp, f"{img_name}_DSM_epoch_{epoch}.tif")
    # find ground truth data
    dsm_tif_fp = dataset.dsm_tif_fp
    metadata_fp = dataset.dsm_txt_fp
    metadata = np.loadtxt(metadata_fp)

    # create dsm and save to output_fp
    create_dsm(lats, lons, alts, output_fp)

    # water masks
    watermask_fp, ignore_mask_fp = None, None
    if dataset.ignore_mask_fp:
        ignore_mask_fp = dataset.ignore_mask_fp
    else:
        watermask_fp = dataset.dsm_cls_fp

    # compute mae
    mae = compute_mae(
        output_fp,
        dsm_tif_fp,
        metadata,
        output_dp,
        gt_water_mask_fp=watermask_fp,
        ignore_mask_fp=ignore_mask_fp,
    )
    return mae


def compute_mae(
    dsm_nerf_fp,
    gt_fp,
    dsm_metadata,
    output_dp,
    gt_water_mask_fp=None,
    ignore_mask_fp=None,
):
    try:
        from osgeo import gdal

        gdal.UseExceptions()
    except ModuleNotFoundError as e:
        import traceback

        traceback.print_exc()
        print(
            "Gdal was not found\n"
            + "Make sure LD_LIBRARY_PATH env variable is set to the conda envs/<name>/lib folder\n"
            + "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/miniconda3/envs/<env-name>/lib"
        )
        exit()

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = os.path.join(
        output_dp, "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    )
    pred_rdsm_path = os.path.join(
        output_dp, "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)
    )
    tmp_gt_path = os.path.join(
        output_dp, "tmp_crop_gt_dsm_to_delete_{}.tif".format(unique_identifier)
    )
    tmp_gt_watermask_path = os.path.join(
        output_dp, "tmp_crop_gt_watermask_to_delete_{}.tif".format(unique_identifier)
    )

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(dsm_nerf_fp)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None

    # do the same to the gt dsm
    ds = gdal.Open(gt_fp)
    ds = gdal.Translate(tmp_gt_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None

    mask = None

    # assert either watermask or ignore mask is used
    assert (
        int(gt_water_mask_fp is None) + int(ignore_mask_fp is None) == 1
    ), "either watermask or ignore mask should be specified for MAE calculation"

    # water masks
    if gt_water_mask_fp and os.path.isfile(gt_water_mask_fp):
        # also crop in water mask
        ds = gdal.Open(gt_water_mask_fp)
        ds = gdal.Translate(tmp_gt_watermask_path, ds, projWin=[ulx, uly, lrx, lry])
        ds = None

        with rasterio.open(tmp_gt_watermask_path, "r") as f:
            water_mask = f.read()[0, :, :]
            mask = water_mask.copy()
            mask[water_mask != 9] = 0
            mask[water_mask == 9] = 1

    if ignore_mask_fp and os.path.isfile(ignore_mask_fp):
        with rasterio.open(ignore_mask_fp, "r") as f:
            mask = f.read(1).squeeze()

    if mask is not None:
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, "w", **profile) as dst:
            pred_dsm[mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    with rasterio.open(tmp_gt_path, "r") as f:
        gt_dsm = f.read()[0, :, :]

        # replace values below a high depth with 0
        gt_dsm[gt_dsm < -500.0] = 0.0

    # with rasterio.open(pred_dsm_path, "r") as f:
    #    profile = f.profile
    #    pred_dsm = f.read()[0, :, :]

    transform = dsmr.compute_shift(tmp_gt_path, pred_dsm_path, scaling=False)
    dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
    with rasterio.open(pred_rdsm_path, "r") as f:
        pred_rdsm = f.read()[0, :, :]

    diff = pred_rdsm - gt_dsm

    # save diff to disk
    dsm_diff_ofp = os.path.join(
        output_dp, os.path.basename(dsm_nerf_fp)[:-4] + "_error.tif"
    )
    with rasterio.open(dsm_nerf_fp) as src:
        profile = src.profile
        with rasterio.open(dsm_diff_ofp, "w", **profile) as dst:
            dst.write(diff, 1)

    # remove temp files
    os.remove(pred_dsm_path)
    os.remove(pred_rdsm_path)
    os.remove(tmp_gt_path)
    if os.path.isfile(tmp_gt_watermask_path):
        os.remove(tmp_gt_watermask_path)

    return {
        "mean": "{:.3f}".format(np.nanmean(abs(diff.ravel()))),
        "median": "{:.3f}".format(np.nanmedian(abs(diff.ravel()))),
    }
