import abc
import framework.util.conversions as conversions


class BaseCoordinateSystem:
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    @abc.abstractmethod
    def from_latlon(self, lat, lon, alts):
        pass

    @abc.abstractmethod
    def to_lat_lon(self, x, y, z):
        pass


class CoordinateSystemCustomECEF(BaseCoordinateSystem):
    def from_latlon(self, lat, lon, alts):
        return conversions.latlon_to_ecef_custom(lat, lon, alts)

    def to_lat_lon(self, x, y, z):
        return conversions.ecef_to_latlon_custom(x, y, z)


class CoordinateSystemUTM(BaseCoordinateSystem):

    def __init__(self, dataset, zone_string=None) -> None:
        super().__init__(dataset)
        self.zone_string = dataset.zone_string if zone_string is None else zone_string

    def from_latlon(self, lat, lon, alts):
        eastings, northings, _ = conversions.utm_from_latlon(
            lat, lon, zone_string=self.zone_string
        )
        return eastings, northings, alts

    def to_lat_lon(self, eastings, northings, alts):
        return (
            *conversions.latlon_from_utm(eastings, northings, self.zone_string),
            alts,
        )
