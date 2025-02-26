from torchvision import transforms as transforms
import rasterio
from PIL import Image
import torch
import os
import numpy as np


def load_tensor_from_cls_geotiff(img_path):
    with rasterio.open(img_path, "r") as f:
        labels = f.read()  # [1, h, w]

    labels = torch.from_numpy(labels)
    labels = labels.view(1, -1).permute(1, 0)  # (h*w, 1)
    # print("labels contain non-zero values: ", labels.any())
    # print("labels.dtype", labels.dtype)
    # labels = labels.to(dtype=torch.int)
    # print("labels contain non-zero values: ", labels.any())
    # print("labels.dtype", labels.dtype)
    return labels


def load_tensor_from_rgb_geotiff(img_path, downscale_factor=1.0, imethod=Image.BICUBIC):
    with rasterio.open(img_path, "r") as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.0
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = transforms.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = transforms.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs


def load_tensor_from_png(img_path, return_alpha=False):
    """
    Load rgb tensor from .png image
    :param img_path: path to the .png
    :param return_alpha: if true, the alpha channel is returned
    :return: rgb tensor, alpha tensor only if return_alpha_mask=true
    """
    img = Image.open(img_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)
    rgb = img_tensor[..., :3]  # cut off alpha channel if one exists

    if return_alpha:
        assert img_tensor.shape[2] == 4, "Image does not contain an alpha channel"
        alpha = img_tensor[..., 3]
        return rgb, alpha

    return rgb


def save_output_image(input, output_path, source_path, copy_rpc=False):
    """
    Save output into a .tif file using rasterio
    :param input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
                    can be a pytorch tensor or a numpy array
    :param output_path: where to store the output
    :param source_path: .tif file that can be referenced for metadata
    :param copy_rpc: should the rpc model of the source_path .tif be copied over
    """
    # convert input to numpy array float32
    # if torch.is_tensor(input):
    #     im_np = input.type(torch.FloatTensor).cpu().numpy()
    # else:
    #     im_np = input.astype(np.float32)
    if torch.is_tensor(input):
        im_np = input.cpu().numpy()
    else:
        im_np = input

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, "r") as src:
        profile = src.profile
        # profile["dtype"] = rasterio.float32
        profile["dtype"] = im_np.dtype
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(im_np)
            if copy_rpc:
                dst.update_tags(ns="RPC", **src.tags(ns="RPC"))
