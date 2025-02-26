import os
from tqdm import tqdm
from lightning.pytorch.loggers import TensorBoardLogger
import cv2
import sys

from framework.visualize import run_visualizer
from eval.utils.util import run_eval_script

import semantic.components.visualize as visualize
import baseline.components.visualize as baseline_visualize


def create_visualizers(cfgs):
    visualizer_semantic_rendering = visualize.SemanticColorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True
    )
    visualizer_semantic_error = visualize.SemanticErrorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True
    )
    visualizer_summary = visualize.TensorboardSemanticSummaryVisualization(
        cfgs, save_as_tif=False, send_to_tensorboard=True
    )
    visualizer_semantic_rendering_shading = visualize.SemanticColorShadingVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True
    )
    rgb = baseline_visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="rgb"
    )
    beta = baseline_visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=False, factor_name="beta"
    )
    sun = baseline_visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=False, factor_name="sun"
    )
    depth = baseline_visualize.FactorVisualization(
        cfgs,
        save_as_tif=True,
        send_to_tensorboard=False,
        factor_name="depth",
        cmap=cv2.COLORMAP_JET,
    )
    albedo = baseline_visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=False, factor_name="albedo"
    )
    beta_s = baseline_visualize.FactorVisualization(
        cfgs, save_as_tif=True, send_to_tensorboard=False, factor_name="beta_semantic"
    )
    confusion_matrix = visualize.ConfusionMatrixVisualization(
        cfgs, save_as_tif=False, send_to_tensorboard=True
    )
    visualizers = [
        rgb,
        beta,
        sun,
        albedo,
        depth,
        # visualizer_summary,
        visualizer_semantic_rendering,
        visualizer_semantic_rendering_shading,
        visualizer_semantic_error,
        confusion_matrix,
    ]
    # visualizers = []

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
