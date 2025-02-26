import numpy as np
import torch
from framework.logger import logger

from framework.components.camera_models import BaseCameraModel
from framework.components.coordinate_systems import BaseCoordinateSystem


def construct_sun_dir(
    sun_elevation_deg: float, sun_azimuth_deg: float, n_rays: int
) -> torch.FloatTensor:
    """
    Create vector encoding the sun direction
    :param sun_elevation_deg: float, sun elevation in  degrees
    :param sun_azimuth_deg: float, sun azimuth in degrees
    :param n_rays: number of rays affected by the same sun direction
    :return: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
    """

    sun_el = np.radians(sun_elevation_deg)
    sun_az = np.radians(sun_azimuth_deg)
    sun_d = np.array(
        [
            np.sin(sun_az) * np.cos(sun_el),
            np.cos(sun_az) * np.cos(sun_el),
            np.sin(sun_el),
        ]
    )
    sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
    sun_dirs = sun_dirs.type(torch.FloatTensor)
    return sun_dirs


def satnerf_construct(
    camera_model: BaseCameraModel,
    coordinate_system: BaseCoordinateSystem,
    rows,
    cols,
    min_alt: float,
    max_alt: float,
):
    rows, cols = rows.flatten(), cols.flatten()
    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons, lats = camera_model.localization(cols, rows, max_alts)
    x_near, y_near, z_near = coordinate_system.from_latlon(lats, lons, max_alts)
    xyz_near = np.vstack([x_near, y_near, z_near]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    lons, lats = camera_model.localization(cols, rows, min_alts)
    x_far, y_far, z_far = coordinate_system.from_latlon(lats, lons, min_alts)
    xyz_far = np.vstack([x_far, y_far, z_far]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the ray's origin, direction vector and near-far bounds
    rays = torch.from_numpy(
        np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]])
    )
    rays = rays.type(torch.FloatTensor)
    return rays
