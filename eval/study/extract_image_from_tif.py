import rasterio
from PIL import Image
import os
import numpy as np
import glob
import cv2
from tqdm import tqdm
from framework.util.other import visualize_image_numpy

"""
This script extracts a .png image from a given .tif
If given a dir, extracts all .tifs inside of the dir and saves the pngs in a child dir
"""


def extract_single(tif_ifp, output_fp, cmap="BONE"):
    with rasterio.open(tif_ifp, "r") as fp:
        img = fp.read()
        img = np.moveaxis(img, 0, -1)

        # replace nan values with lowest point
        img = np.nan_to_num(img, nan=np.nanmin(img))

        mean = np.nanmean(img)
        # print("Image mean:", mean)
        # print("Image shape:", img.shape)

        # if the mean is less than 0, we need to convert image to [0, 255] range
        if mean < 1:
            img = (255 * (img - np.nanmin(img)) / np.ptp(img)).astype(np.uint8)
            # print("Scaling image to [0, 255]")
            # print("Image mean after scaling:", np.nanmean(img))
        # img = img.astype(np.uint8)
        # print(img.shape)

        if img.shape[2] == 1:
            img = np.squeeze(img)
            # apply color map to it
            img = visualize_image_numpy(
                img, getattr(cv2, "COLORMAP_" + cmap, cv2.COLORMAP_BONE)
            )
            # print("Applied cv2 color map")
            # print("Image mean after color map:", np.nanmean(img))
            # print("Image shape:", img.shape)

        pil_img = Image.fromarray(img)
        pil_img.save(output_fp)


def extract(input_path, cmap="BONE"):

    if os.path.isdir(input_path):
        output_dp = os.path.join(
            os.path.dirname(input_path), os.path.basename(input_path) + "_png"
        )
        os.makedirs(output_dp, exist_ok=True)
        print(f"Saving extracted pngs to: {output_dp}")
        files = glob.glob(os.path.join(input_path, "*.tif"))
        files = sorted(files)
        for file in tqdm(files):
            extract_single(
                file, os.path.join(output_dp, os.path.basename(file)[:-4] + ".png")
            )
    else:
        output_fp = input_path[:-4] + ".png"
        extract_single(input_path, output_fp, cmap)
        print(f"Saved extracted png to: {output_fp}")


if __name__ == "__main__":
    import fire

    fire.Fire(extract)
