import abc
import os

import numpy as np
import cv2
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from lightning.pytorch.loggers import TensorBoardLogger
import sys

import framework.util.img_utils as img_utils
from framework.util.other import (
    scale_image_for_tensorboard,
    visualize_image,
    w_h_from_sample,
)
from framework.logger import logger as logger_text
from framework.configs import load_configs_from_logs, adapt_configs_for_inference
from framework.util.load_ckpoint import load_from_disk
from eval.utils.util import batched_inference


class BaseVisualization:
    def __init__(self, cfgs, send_to_tensorboard: bool) -> None:
        super().__init__()
        self.cfgs = cfgs
        self.send_to_tensorboard = send_to_tensorboard

    @abc.abstractmethod
    def run(
        self,
        pipeline,
        dataset,
        sample,
        results,
        sample_idx: int = 0,
        split: str = "test",
        epoch: int = 0,
        source_fp: str = None,
        logger=None,
        force_output_fp=None,
    ):
        pass


class ImageVisualization(BaseVisualization):

    def __init__(self, cfgs, send_to_tensorboard: bool, save_as_tif: bool) -> None:
        super().__init__(cfgs, send_to_tensorboard)
        self._save_as_tif = save_as_tif

    def run(
        self,
        pipeline,
        dataset,
        sample,
        results,
        sample_idx: int = 0,
        split: str = "test",
        epoch: int = 0,
        source_fp: str = None,
        logger=None,
        force_output_fp=None,
    ):
        # logger_text.info("Vis", f"Running visualization: {self._name()}")

        vis_output, W, H = self.visualize(pipeline, dataset, sample, results)

        if vis_output is None:
            # visualize indicates it does not have anything to show for the given input
            return

        if self.send_to_tensorboard and logger is not None:
            img = self._visualize_image_for_tensorboard(vis_output, W, H)

            # tensorboard expects NCHW format
            if len(img.shape) == 3:
                img = img[None, :]

            logger.experiment.add_images(
                f"{split}_{sample_idx}/{self._name()}", img, epoch
            )

        if self._save_as_tif:
            # to save as .tif it needs to be in (C, H, W)
            if len(vis_output.shape) == 2:
                # if single band, expand to have 1-channel at the front
                vis_output = vis_output[None, :]

            if type(sample["name"]) is list:
                name = sample["name"][0]
            else:
                name = sample["name"]

            self._save(
                vis_output,
                name,
                split=split,
                epoch=epoch,
                source_fp=source_fp,
                force_output_fp=force_output_fp,
            )

    def visualize(self, pipeline, dataset, sample, results):
        typ = ""
        if "rgb_fine" in results:
            typ = "_fine"
        if "rgb_coarse" in results:
            typ = "_coarse"
        W, H = w_h_from_sample(sample)
        viz_output = self._visualize(pipeline, dataset, sample, results, W, H, typ)

        if viz_output is None:
            # visualize indicates it does not have anything to show for the given input
            return None, None, None

        # the channels need to be [C, W, H]
        # if [W, H, C], the scaling for tensorboard takes forever until crashing
        if len(viz_output.shape) == 3:
            assert viz_output.shape[0] in [
                1,
                3,
                4,
            ], "Wrong channel order in visualization. Needs to be [C, W, H]"
        return viz_output, W, H

    @abc.abstractmethod
    def _visualize(self, pipeline, dataset, sample, results, W, H, typ):
        pass

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # if the image has only one channel, we have to visualize the output

        if len(img.shape) == 2:
            img = visualize_image(
                img,
                cmap=self._get_visualize_color_scheme(),
                cmap_bounds=self._get_visualize_color_range(),
            )
            img = img[None, :]

        # scale image down so that it fits into tensorboard UI
        return scale_image_for_tensorboard(img)

    def visualize_image_cmap_and_save(
        self, pipeline, dataset, sample, results, save_to_fp
    ):
        typ = "_fine" if "rgb_fine" in results else "_coarse"
        W, H = w_h_from_sample(sample)
        img = self._visualize(pipeline, dataset, sample, results, W, H, typ)
        if len(img.shape) == 2:
            img = visualize_image(img, cmap=self._get_visualize_color_scheme())
            img = img[None, :]
        if img.max() > 1:
            img = img.to(torch.float32)
            img /= 255.0
        save_image(img, save_to_fp)

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_JET

    def _get_visualize_color_range(self):
        return None

    @abc.abstractmethod
    def _name(self) -> str:
        pass

    def _save(
        self,
        visualize_output,
        name: str,
        split: str = "test",
        epoch: int = 0,
        source_fp: str = None,
        force_output_fp=None,
    ):
        output_fp = os.path.join(
            self.cfgs.run.run_dp,
            "visualization",
            split,
            self._name(),
            f"{name}_{epoch}.tif",
        )

        if force_output_fp is not None:
            output_fp = force_output_fp

        img_utils.save_output_image(
            visualize_output,
            output_path=output_fp,
            source_path=source_fp,
            copy_rpc=True,
        )


def run_visualizer(
    input_dp: str,
    output_dp: str = None,
    split="test",
    epoch=-1,
    device=0,
    device_req_free=True,
    create_visualizers_fn=None,
    tensorboard=False,
    save_png=True,
    max_items=1000000,
    render_options_fn=lambda pipeline, split: pipeline._val_render_options(split),
):
    """
    The purpose of this script is to allow running of new visualizers on an existing, trained model.
    The outputs are stored in the model folder under visualizations, as if it has existed during training
    This script is extended by each sub-repository to use the visualizers defined in each repo.
    :param input_dp: input to a model
    :param output_dp: not used.
    :param split: train|test
    :param epoch: which .ckpt to load
    :param device: which cuda device if multiple exist
    :param device_req_free: require cuda device to be free
    :param create_visualizers_fn: function(cfgs) -> list of implementations of framework.components.BaseVisualizer
    :param tensorboard: if true, send results of visualizers to tensorboard of model
    :param save_png: if true, export results as png in addition to the tif
    :param max_items: how many images are evaluated at maximum
    :param render_options_fn: set render options applied to inference
    """
    assert os.path.isdir(input_dp), "log_dp is not a path to an existing folder"
    assert create_visualizers_fn is not None and callable(
        create_visualizers_fn
    ), "create_visualizers_fn needs to be set to a function returning the visualizers that should be run"

    cfgs = load_configs_from_logs(input_dp)
    cfgs = adapt_configs_for_inference(cfgs)

    # load trained nerf
    models, pipeline, epoch, cuda_device = load_from_disk(
        cfgs, input_dp, epoch, device, device_req_free
    )

    # load the datasets
    pipeline.load_datasets()

    dataset_name = "rgb"
    if split == "test":
        dataset_name = "rgb_test"

    dataset = pipeline.datasets[dataset_name]
    dataset.force_act_as_test()

    logger = TensorBoardLogger(
        save_dir=cfgs.run.run_dp,
        name=None,
        default_hp_metric=False,
        version="tensorboard",  # sets the folder name for the logs
    )

    visualizers = create_visualizers_fn(cfgs)
    send_to_tensorboard_wished = tensorboard
    for visualizer in visualizers:
        # disable sending to tensorboard for visualizers if turned off through cmd line args
        visualizer.send_to_tensorboard = (
            visualizer.send_to_tensorboard and send_to_tensorboard_wished
        )

    render_options = render_options_fn(pipeline, split)

    until = min(len(dataset), max_items)
    for img_idx in tqdm(range(until)):
        img = dataset[img_idx]
        rays = img["rays"].to(cuda_device)
        extras = img["extras"].to(cuda_device)

        results = batched_inference(
            cfgs, pipeline.renderer, models, rays, extras, render_options=render_options
        )

        for visualizer in visualizers:
            img_split = split
            sample_idx = img_idx
            if split == "test":
                if img_idx == 0:
                    # the first image of the test split is the first train image
                    img_split = "train"
                else:
                    # the index for the rest of the test images needs to be decreased by one
                    sample_idx = img_idx - 1

            if save_png:
                # save the visualizer output as png instead
                # this is kinda hacky but it works for the few cases it is wanted
                output_dp = os.path.join(
                    cfgs.run.run_dp,
                    "visualization",
                    img_split,
                    visualizer._name(),
                )
                os.makedirs(output_dp, exist_ok=True)
                output_fp = os.path.join(output_dp, f"{img['name']}_{epoch}.png")
                visualizer.visualize_image_cmap_and_save(
                    pipeline, dataset, img, results, save_to_fp=output_fp
                )

            visualizer.run(
                pipeline,
                dataset,
                img,
                results,
                sample_idx=sample_idx,
                split=img_split,
                epoch=epoch,
                source_fp=img["img_fp"],
                logger=logger,
            )
