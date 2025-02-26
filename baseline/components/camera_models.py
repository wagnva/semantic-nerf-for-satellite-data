import numpy as np
from rpcm import RPCModel
import rpcm
import torch

from framework.components.camera_models import BaseCameraModel
import framework.util.sat_utils as sat_utils
from framework.components.coordinate_systems import BaseCoordinateSystem
from framework.components.normalization import BaseNormalization


class CameraModelRPC(BaseCameraModel):
    def __init__(self, rpc: RPCModel) -> None:
        super().__init__()
        self._rpc = rpc

    def localization(self, cols, rows, alts):
        return self._rpc.localization(cols, rows, alts)

    def projection(self, lon, lat, alts):
        return self._rpc.projection(lon, lat, alts)


def construct_rpc_camera_model(d: dict, scale_factor=1.0) -> CameraModelRPC:
    """
    This method encapsulates construction of the rpc camera model instance
    :param d: dictionary containing the metadata from the .json files for each view
    :param scale_factor: scale the rpc model
    :return: rpc camera modell instance
    """
    rpc = sat_utils.rescale_rpc(
        rpcm.RPCModel(d["rpc"], dict_format="rpcm"),
        1.0 / scale_factor,
    )

    return CameraModelRPC(rpc)
