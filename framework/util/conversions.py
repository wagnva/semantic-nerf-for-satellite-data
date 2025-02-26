import numpy as np
import utm
import pymap3d
import torch


def convert_utm_to_local(dataset, utm_points: np.ndarray):
    """
    Converts points in unnormalized UTM coordinate space to the local (either utm or ecef) normalized coordinate system
    :param dataset: BaseDataset
    :param utm_points: Unnormalized UTM numpy Array (N_points, 3)
    :return: normalized local numpy array (N_points, 3)
    """
    # convert to lat lon first
    lat, lon, alts = (
        *latlon_from_utm(utm_points[:, 0], utm_points[:, 1], dataset.zone_string),
        utm_points[:, 2],
    )
    # to the local coordinate system
    x, y, z = dataset.coordinate_system.from_latlon(lat, lon, alts)
    xyz = np.stack([x, y, z], axis=1)  # .T.reshape((-1, 3))
    # normalize
    xyz_n = dataset.normalization_component.normalize_xyz(torch.tensor(xyz))
    return xyz_n.numpy()


def convert_local_to_utm(dataset, xyz: np.ndarray):
    """
    Converts points in normalized local coordinate system (either utm or ecef) into unnormalized UTM
    :param dataset: BaseDataset
    :param xyz: normalized local numpy Array (N_points, 3)
    :return: unnormalized utm numpy array (N_points, 3)
    """
    xyz_un = dataset.normalization_component.denormalize({"xyz": torch.from_numpy(xyz)})
    lat, lon, alts = dataset.coordinate_system.to_lat_lon(
        xyz_un[:, 0], xyz_un[:, 1], xyz_un[:, 2]
    )
    x, y, _ = utm_from_latlon(lat.numpy(), lon.numpy(), zone_string=dataset.zone_string)
    xyz = np.vstack([x, y, alts]).T
    return xyz


def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z


def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a**2
    esq = e**2
    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x**2) + (y**2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + (ep**2) * b * (np.sin(th) ** 3)),
        (p - esq * a * (np.cos(th) ** 3)),
    )
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


# def utm_from_latlon(lats, lons):
#     """
#     convert lat-lon to utm
#     """
#     import pyproj
#     import utm
#     from pyproj import Transformer
#
#     n = utm.latlon_to_zone_number(lats[0], lons[0])
#     l = utm.latitude_to_zone_letter(lats[0])
#     proj_src = pyproj.Proj("+proj=latlong")
#     proj_dst = pyproj.Proj("+proj=utm +zone={}{}".format(n, l))
#     transformer = Transformer.from_proj(proj_src, proj_dst)
#     easts, norths = transformer.transform(lons, lats)
#     # easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
#     return easts, norths


def utm_from_lonlat(lons, lats, zone_string=None):
    """
    convert lon-lat to utm
    """
    return utm_from_latlon(lats, lons, zone_string)


def utm_from_latlon(lats, lons, zone_string=None):
    """
    convert lat-lon to utm
    """
    if zone_string is None:
        result = utm.from_latlon(lats, lons)
    else:
        result = utm.from_latlon(lats, lons, *split_zone_string(zone_string))

    return result[0], result[1], str(result[2]) + result[3]


def latlon_from_utm(easts, norths, zone_string):
    """
    convert utm to lat lon
    """
    return utm.to_latlon(easts, norths, *split_zone_string(zone_string))


def lonlat_from_utm(easts, norths, zone_string):
    """
    convert utm to lon-lat
    """
    results = latlon_from_utm(easts, norths, zone_string)
    return (
        results[1],
        results[0],
    )  # flip since the utm method returns latlon, but we want to return lonlat


def split_zone_string(zone_string):
    return int(zone_string[:-1]), zone_string[-1]


def zonestring_to_hemisphere(zonestring):
    zone_number, zone_letter = split_zone_string(zonestring)
    if zone_letter >= "N":
        return str(zone_number) + "N"
    else:
        return str(zone_number) + "S"


def enu_to_latlonalt(e, n, u, lat0, lon0, alt0):
    # Source: https://github.com/Kai-46/VisSatSatelliteStereo/blob/c6cb1b4ca6bfc6f7210707333db3bbd8931a6265/lib/latlonalt_enu_converter.py#L42
    lat, lon, alt = pymap3d.enu2geodetic(e, n, u, lat0, lon0, alt0)
    return lat, lon, alt


def latlonalt_to_enu(lats, lons, alts, lat0, lon0, alt0):
    x, y, z = pymap3d.geodetic2enu(lats, lons, alts, lat0, lon0, alt0)
    return x, y, z


def qvec2rotmat(qvec):
    # source: https://github.com/colmap/colmap/blob/b59256d89b206d35d517e26d077d51fcd18d48b8/scripts/python/read_write_model.py#L524
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )
