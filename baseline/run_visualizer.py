import os
from tqdm import tqdm
from lightning.pytorch.loggers import TensorBoardLogger
import cv2
import sys

from framework.visualize import run_visualizer
from eval.utils.util import run_eval_script

import baseline.components.visualize as visualize


def create_visualizers(cfgs):
    visualizer_diff = visualize.RGBDiffVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True
    )
    visualizer_diff_distance = visualize.RGBDiffDistanceVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True
    )
    visualizer_uncertainty = visualize.FactorVisualization(
        cfgs,
        save_as_tif=True,
        send_to_tensorboard=True,
        factor_name="beta",
        cmap=cv2.COLORMAP_BONE,
    )
    visualizer_rgb = visualize.FactorVisualization(
        cfgs, save_as_tif=False, send_to_tensorboard=True, factor_name="rgb"
    )
    visualizer_alts = visualize.AltsVisualization(
        cfgs, save_as_tif=False, send_to_tensorboard=False
    )
    visualizer_depths = visualize.FactorVisualization(
        cfgs, save_as_tif=False, send_to_tensorboard=False, factor_name="depth"
    )
    visualizer_irradiance = visualize.FactorVisualization(
        cfgs,
        save_as_tif=True,
        send_to_tensorboard=True,
        factor_name="irradiance",
    )
    visualizer_sky = visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="sky"
    )
    visualizer_sun = visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="sun"
    )
    visualizer_summary = visualize.TensorboardSummaryVisualization(
        cfgs,
        save_as_tif=False,
        send_to_tensorboard=True,
    )

    visualizers = [visualizer_rgb, visualizer_alts, visualizer_depths, visualizer_sun]

    return visualizers


if __name__ == "__main__":

    import fire

    fire.Fire(
        lambda input_dp, *args, **kwargs: run_eval_script(
            run_visualizer,
            input_dp,
            input_dp,
            *args,
            **kwargs,
            create_visualizers_fn=create_visualizers
        )
    )
