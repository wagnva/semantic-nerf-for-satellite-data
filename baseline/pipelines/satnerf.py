import torch
import numpy as np
from typing import Union
import cv2

from framework.components.rendering import BaseRenderer
from baseline.pipelines.base_ray_pipeline import BaseRayPipeline
from framework.visualize import BaseVisualization
from framework.components.training_step import BaseTrainingStep

from baseline.components.training_step import SatNeRFTrainingStep
from baseline.dataset.satnerf_dataset import SatNeRFDataset
from baseline.dataset.satnerf_depth_dataset import (
    SatNeRFDepthDataset,
)
from baseline.models.satnerf import SatNeRF
from baseline.components.rendering import SatNeRFRendering
from baseline.components.loss import SNerfLoss, SatNerfLoss, DepthLoss
import baseline.components.visualize as visualize
from baseline.pipelines.snerf import SNeRFConfig


class SatNeRFPipeline(BaseRayPipeline):
    def __init__(self, cfgs, ckpt_info=None) -> None:
        super().__init__(cfgs, ckpt_info)
        if self.cfgs.pipeline.depth_enabled:
            self.ds_drop = np.round(
                self.cfgs.pipeline.depth_supervision_drop * self.cfgs.run.max_train_steps
            )
            self.logger_text.info(
                "Depth",
                f"During Training Depth Supervision is used until iteration {self.ds_drop}",
            )

    def _init_datasets(self) -> dict:
        d = {
            "rgb": SatNeRFDataset(self.cfgs, "rgb", "train"),
            "rgb_test": SatNeRFDataset(self.cfgs, "rgb", "test"),
        }
        if self.cfgs.pipeline.depth_enabled:
            d["depth"] = SatNeRFDepthDataset(self.cfgs, "depth", "train")
        return d

    def _init_loss(self):
        self.loss = SatNerfLoss(lambda_sc=self.cfgs.pipeline.sc_lambda)
        self.loss_without_beta = SNerfLoss(lambda_sc=self.cfgs.pipeline.sc_lambda)
        if self.cfgs.pipeline.depth_enabled:
            # depth supervision will be used
            self.depth_loss = DepthLoss(lambda_ds=self.cfgs.pipeline.ds_lambda)

    def _init_models(self) -> dict:
        d = {
            "coarse": SatNeRF(
                self.cfgs,
                layers=self.cfgs.pipeline.fc_layers,
                feat=self.cfgs.pipeline.fc_units,
                skips=self.cfgs.pipeline.fc_skips,
                t_embedding_dims=self.cfgs.pipeline.t_embedding_tau,
            ),
            "t": torch.nn.Embedding(
                self.cfgs.pipeline.t_embedding_vocab,
                self.cfgs.pipeline.t_embedding_tau,
            ),
        }

        return d

    def _init_renderer(self) -> BaseRenderer:
        return SatNeRFRendering(self.cfgs)

    def _init_training_step(self) -> BaseTrainingStep:
        return SatNeRFTrainingStep()

    def _init_visualizers(self) -> list[BaseVisualization]:
        return [
            visualize.TensorboardSummaryVisualization(
                self.cfgs, save_as_tif=False, send_to_tensorboard=True
            ),
            visualize.FactorVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="rgb"
            ),
            visualize.FactorVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="depth"
            ),
            visualize.FactorVisualization(
                self.cfgs,
                save_as_tif=True,
                send_to_tensorboard=True,
                factor_name="albedo",
            ),
            visualize.FactorVisualization(
                self.cfgs,
                save_as_tif=True,
                send_to_tensorboard=True,
                factor_name="sun",
                cmap=cv2.COLORMAP_BONE,
            ),
            visualize.FactorVisualization(
                self.cfgs,
                save_as_tif=True,
                send_to_tensorboard=True,
                factor_name="beta",
                cmap=cv2.COLORMAP_BONE,
            ),
            visualize.RGBDiffDistanceVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True
            ),
        ]

    @classmethod
    def init_config(cls, cfg_information):
        return SatNeRFConfig(**cfg_information)


class SatNeRFConfig(SNeRFConfig):
    fc_use_full_features: Union[bool, int] = False

    depth_enabled: Union[bool, int] = True
    depth_supervision_drop: float = 0.25
    ds_lambda: int = 1000
    first_beta_epoch: int = 2
    t_embedding_vocab: int = 50
    t_embedding_tau: int = 4
    ds_noweights: Union[bool, int] = False

    def determine_run_name_postfix(self):
        postfix = super().determine_run_name_postfix()
        if not self.depth_enabled:
            postfix += "_no_ds"
        if self.t_embedding_tau != 4:
            postfix += f"_t{self.t_embedding_tau}"
        return postfix
