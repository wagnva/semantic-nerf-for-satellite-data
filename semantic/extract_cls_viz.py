import rasterio
import torch
from PIL import Image
import numpy as np

from semantic.components.visualize import get_semantic_class_color_mapping


def extract(tif_ifp):
    with rasterio.open(tif_ifp, "r") as fp:
        img = fp.read()
        img = np.moveaxis(img, 0, -1)  # (H, W, 1)
        img = img.squeeze()  # (H, W)

        mapping = get_semantic_class_color_mapping().numpy()
        img = mapping[img]

        pil_img = Image.fromarray(img)
        output_fp = tif_ifp[:-4] + ".png"
        pil_img.save(output_fp)


if __name__ == "__main__":
    import fire

    fire.Fire(extract)
