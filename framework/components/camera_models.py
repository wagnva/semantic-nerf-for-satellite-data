import abc


class BaseCameraModel:
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def localization(self, cols, rows, max_alts):
        pass

    @abc.abstractmethod
    def projection(self, lon, lat, alts):
        pass
