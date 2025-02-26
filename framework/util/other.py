import numpy
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
from framework.logger import logger


SCALE_IMAGE_WIDTH_PIXELS_SMALL = 400
SCALE_IMAGE_WIDTH_PIXELS_NORMAL = 600


def scale_image_for_tensorboard(
    img: torch.tensor, size=SCALE_IMAGE_WIDTH_PIXELS_NORMAL
) -> torch.tensor:
    """
    Scale torch tensor to fixed width with variable height based on aspect ratio
    :param img: torch tensor with shape (3, W, H)
    :param size: length of longest side
    :return: rescaled image with shape (3, SCALE_IMAGE_WIDTH_PIXELS, H')
    """
    resize = torchvision.transforms.Resize(size=size, antialias=True)
    return resize(img)


def visualize_image_numpy(x: np.ndarray, cmap=cv2.COLORMAP_JET, cmap_bounds=None):
    x = np.nan_to_num(x)  # change nan to 0
    if cmap_bounds is None:
        mi = np.min(x)
        ma = np.max(x)
    else:
        mi, ma = cmap_bounds
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x = np.clip(x, 0, 255)
    return cv2.applyColorMap(x, cmap)


def visualize_image(img: torch.tensor, cmap=cv2.COLORMAP_JET, cmap_bounds=None):
    """
    applies colormap to an image
    :param img: image (H, W)
    :param cmap: the colormap to apply to the image
    :param cmap_bounds: (min, max) range of values to use for normalization. If not set, min()/max() vals are used
    :return: colored image (H, W, 3)
    """
    x = img.squeeze().cpu().numpy()  # (H, W)
    x = visualize_image_numpy(x, cmap, cmap_bounds)
    x_ = Image.fromarray(x)
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_


def w_h_from_sample(sample):
    if "h" in sample and "w" in sample:
        if type(sample["w"]) is list:
            W, H = sample["w"][0], sample["h"][0]
        else:
            W, H = sample["w"], sample["h"]
    else:
        W = H = int(
            torch.sqrt(torch.tensor(sample["rays"].squeeze().shape[0]).float())
        )  # assume squared images
    return int(W), int(H)
