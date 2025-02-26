import torch
import os

from framework.logger import logger


def _satnerf_ray_component(rays: torch.tensor, name: str, value=None):
    """
    This method allows named access to the different dimensions of the ray tensor
    :param rays: tensor containing rays
        expected: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
    :param name: name of the component
    :param value: if given, this method overwrites the value at rays using the given name
    :return: the columns corresponding to the given name
    """
    start, end = -1, -1
    if name.startswith("origin"):
        start, end = 0, 3
    elif name.startswith("dir"):
        start, end = 3, 6
    elif name.startswith("near"):
        start, end = 6, 7
    elif name.startswith("far"):
        start, end = 7, 8
    elif name.startswith("sun_direction"):
        start, end = 8, 11
    else:
        logger.error(
            "Dataset", f"Trying to access ray component with a unknown name: {name}"
        )

    if value is not None:
        rays[:, start:end] = value

    return rays[:, start:end]


def _satnerf_extras_component(extras: torch.tensor, name: str, value=None):
    """
    This method allows named access to the different dimensions of the extras tensor
    This contains things that are not equal between all nerf variants, such as sun directions, timestamp etc.
    :param extras: tensor containing extras
        expected: (h*w, ?) tensor of floats encoding h*w extras for h*w rays
    :param name: name of the component
    :param value: if given, this method overwrites the value at rays using the given name
    :return: the columns corresponding to the given name
    """
    start, end = -1, -1
    if name.startswith("sun_d"):
        start, end = 0, 3
    elif name.startswith("ts"):
        start, end = 3, 4
    else:
        logger.error(
            "Dataset", f"Trying to access extra component with a unknown name: {name}"
        )

    if value is not None:
        extras[:, start:end] = value

    return extras[:, start:end]


# mask the specific function in case i want to switch it out later on
# maybe set this somewhere based on the config or so?
# at the moment waiting until I need it set to something else
ray_component_fn = _satnerf_ray_component
extras_component_fn = _satnerf_extras_component


def save_to_disk(data: torch.tensor, name: str, cache_dp: str):
    torch.save(data, os.path.join(cache_dp, f"{name}.data"))


def load_from_disk(name: str, cache_dp: str):
    return torch.load(os.path.join(cache_dp, f"{name}.data"), weights_only=True)
