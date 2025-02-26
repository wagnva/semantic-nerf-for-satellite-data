import os
import glob
import rasterio
import numpy as np

from eval.extract_pointcloud import save_ply


def start(path, path_txt_opt=None):
    if path_txt_opt is None:
        dsm_tif_fp = glob.glob(os.path.join(path, "*_DSM.tif"))[0]
        dsm_txt_fp = glob.glob(os.path.join(path, "*_DSM.txt"))[0]
    else:
        dsm_tif_fp = path
        dsm_txt_fp = path_txt_opt
    study(dsm_tif_fp, dsm_txt_fp)


def study(dsm_tif_fp, dsm_txt_fp):

    with rasterio.open(dsm_tif_fp, "r") as fp:
        dsm = fp.read(1)
    geo_vals = read_aoi_txt(dsm_txt_fp)

    # create pointcloud from dsm
    points = create_pc(dsm, geo_vals)
    # create colors based on z-height using cmap viz
    colors = np.ones_like(points)
    # save as ply
    output_fp = dsm_tif_fp[:-4] + ".ply"
    save_ply(points, colors, output_fp)
    print("Saved .ply to:", output_fp)


def create_pc(dsm, geo_vals):
    h, w = dsm.shape
    ulx, lry, xsize, ysize = geo_vals
    xstep = xsize / w
    ystep = ysize / h

    points = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            z = dsm[h - y - 1, x]  # its somehow flipped? this works
            points[y, x] = [ulx + x * xstep, lry + y * ystep, z]

    return points.reshape(-1, 3)


def read_aoi_txt(txt_ifp):
    roi = np.loadtxt(txt_ifp)
    xoff, yoff, xsize, ysize, resolution = (
        roi[0],
        roi[1],
        int(roi[2]),
        int(roi[2]),
        roi[3],
    )
    ulx, uly, lrx, lry = (
        xoff,
        yoff + ysize * resolution,
        xoff + xsize * resolution,
        yoff,
    )
    return ulx, lry, xsize * resolution, ysize * resolution


if __name__ == "__main__":
    import fire

    fire.Fire(start)
