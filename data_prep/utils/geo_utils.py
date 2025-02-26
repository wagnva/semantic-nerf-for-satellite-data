import utm
import numpy as np
import rasterio
import rpcm


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


def lonlat_from_utm(easts, norths, zone_string):
    """
    convert utm to lon-lat
    """
    results = utm.to_latlon(easts, norths, *split_zone_string(zone_string))
    return (
        results[1],
        results[0],
    )  # flip since the utm method returns latlon, but we want to return lonlat


def split_zone_string(zone_string):
    return int(zone_string[:-1]), zone_string[-1]


def read_geojson_polygon_from_txt(txt_ifp, zone_string):
    """
    Read in lonlats and convert into geojson polygon from txt containing scene boundaries.
    Format of txt as written by dataset creation script
    :param txt_ifp: path to txt file
    :param zone_string: utm zone
    :return: geojson polygon
    """
    lons, lats = read_aoi_txt(txt_ifp, return_utm=False, zone_string=zone_string)
    lonlat_bbx = geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    """
    from shapely.geometry import Polygon

    # first attempt to construct the polygon, assuming the input coords_array are ordered
    # the centroid is computed using shapely.geometry.Polygon.centroid
    # taking the mean is easier but does not handle different densities of points in the edges
    pp = coords_array.tolist()
    poly = Polygon(pp)
    x_c, y_c = np.array(poly.centroid.xy).ravel()

    # check that the polygon is valid, i.e. that non of its segments intersect
    # if the polygon is not valid, then coords_array was not ordered and we have to do it
    # a possible fix is to sort points by polar angle using the centroid (anti-clockwise order)
    if not poly.is_valid:
        pp.sort(key=lambda p: np.arctan2(p[0] - x_c, p[1] - y_c))

    # construct the geojson
    geojson_polygon = {"coordinates": [pp], "type": "Polygon"}
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon


def create_affine_transform_from_aoi_txt(txt_ifp):
    bbox = np.loadtxt(txt_ifp)
    ul_e, ul_n, size, resolution = bbox
    ul_n += size * resolution  # necessary to fix rotation
    transform = rasterio.Affine(resolution, 0.0, ul_e, 0.0, -resolution, ul_n)
    return transform


def read_aoi_txt(txt_ifp, return_utm=True, return_size=False, zone_string=None):
    """
    Read in lonlats from txt containing scene boundaries.
    Format of txt as written by dataset creation script
    :param txt_ifp: path to txt file
    :param return_utm: if true, return utm, else return lat/lon
    :param return_size: if true, return the x/y size as well
    :param zone_string: utm zone, required for conversion to lat/lon
    :return: easts,norths / lons, lats
    """
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
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry
    easts = np.array([xmin, xmin, xmax, xmax, xmin])
    norths = np.array([ymin, ymax, ymax, ymin, ymin])

    if return_utm:
        if return_size:
            return easts, norths, xsize * resolution, ysize * resolution
        return easts, norths

    assert (
        zone_string is not None
    ), "zone_string required for conversion from utm to lat/lon"
    lons, lats = lonlat_from_utm(easts, norths, zone_string)

    if return_size:
        return lons, lats, xsize * resolution, ysize * resolution
    return lons, lats


def geojson_to_shapely_polygon(geojson_polygon):
    """
    convert a polygon from geojson format to shapely format
    """
    from shapely.geometry import shape

    return shape(geojson_polygon)


def crop_geotiff_lonlat_aoi(geotiff_path, output_path, lonlat_aoi):
    with rasterio.open(geotiff_path, "r") as src:
        profile = src.profile
        tags = src.tags()
        crs = src.gcps[1]

    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x
    not_pan = len(crop.shape) > 2

    height = crop.shape[0]
    width = crop.shape[1]
    if not_pan:
        height = crop.shape[1]
        width = crop.shape[2]
    else:
        profile["count"] = 1

    profile["height"] = height
    profile["width"] = width
    profile["crs"] = crs

    # fix nan values in the crop
    # replace with 0 -> black
    crop = np.nan_to_num(crop, nan=0.0)

    with rasterio.open(output_path, "w", **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns="RPC", **rpc.to_geotiff_dict())

    return width, height
