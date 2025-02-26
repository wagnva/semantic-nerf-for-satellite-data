import torch
import cv2
from typing import Union, Literal

from framework.visualize import BaseVisualization
from framework.components.training_step import BaseTrainingStep
from framework.components.rendering import BaseRenderer

from baseline.components.training_step import SatNeRFTrainingStep
import baseline.components.visualize as visualize
from baseline.pipelines.satnerf import SatNeRFPipeline, SatNeRFConfig
from baseline.dataset.satnerf_depth_dataset import SatNeRFDepthDataset

from semantic.dataset.semantic_dataset import SemanticDataset
from semantic.components.loss import (
    SemanticLoss,
    SemanticUncertaintyLoss,
    SemanticCarRegLoss,
)
from semantic.components.training_step import RSSemanticTrainingStep
from semantic.components.rendering import RSSemanticRendering
import semantic.components.visualize as visualize_semantic
from semantic.models.rs_semantic import RSSemanticNeRF, inference


class RSSemanticPipeline(SatNeRFPipeline):

    def __init__(self, cfgs, ckpt_info=None):
        super().__init__(cfgs, ckpt_info)
        if cfgs.pipeline.use_tj_instead_of_beta:
            # hacky way to disable beta loss to prevent modification of original training step
            cfgs.pipeline.first_beta_epoch = 10000000

    def _init_datasets(self) -> dict:
        d = {
            "rgb": SemanticDataset(self.cfgs, "rgb", "train"),
            "rgb_test": SemanticDataset(self.cfgs, "rgb", "test"),
        }
        if self.cfgs.pipeline.depth_enabled:
            d["depth"] = SatNeRFDepthDataset(self.cfgs, "depth", "train")
        return d

    def _init_loss(self):
        super()._init_loss()
        self.semantic_loss = SemanticLoss(
            self.cfgs.pipeline.lambda_s,
            self.datasets["rgb"].car_cls_idx,
            ignore_car_index=self.cfgs.pipeline.ignore_car_index,
        )
        self.uncertainty_semantic_loss = SemanticUncertaintyLoss(
            self.cfgs.pipeline.lambda_s,
            self.datasets["rgb"].car_cls_idx,
            detach_beta_for_s=self.cfgs.pipeline.detach_beta_for_s,
            ignore_car_index=self.cfgs.pipeline.ignore_car_index,
        )
        if self.cfgs.pipeline.use_car_reg_loss:
            self.car_reg_loss = SemanticCarRegLoss(
                self.cfgs.pipeline.lambda_c, self.datasets["rgb"].car_cls_idx
            )

    def _init_renderer(self) -> BaseRenderer:
        return RSSemanticRendering(self.cfgs, inference=inference)

    def _init_models(self) -> dict:
        d = {
            "coarse": RSSemanticNeRF(self.cfgs, self.datasets["rgb"]),
            "t": torch.nn.Embedding(
                self.cfgs.pipeline.t_embedding_vocab,
                self.cfgs.pipeline.t_embedding_tau,
            ),
        }

        if self.cfgs.pipeline.use_separate_tj_for_semantic:
            d["t_s"] = torch.nn.Embedding(
                self.cfgs.pipeline.t_embedding_vocab,
                self.cfgs.pipeline.t_embedding_tau,
            )

        return d

    def _init_training_step(self) -> BaseTrainingStep:
        return RSSemanticTrainingStep()

    def _val_render_options(self, split):
        return {}

    def _init_visualizers(self) -> list[BaseVisualization]:
        viz = super()._init_visualizers()
        viz += [
            visualize_semantic.SemanticColorVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=False
            ),
            visualize_semantic.SemanticErrorVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=False
            ),
            visualize_semantic.TensorboardSemanticSummaryVisualization(
                self.cfgs, save_as_tif=False, send_to_tensorboard=True
            ),
            visualize_semantic.SemanticColorShadingVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True
            ),
            visualize_semantic.ConfusionMatrixVisualization(
                self.cfgs, save_as_tif=False, send_to_tensorboard=True
            ),
            visualize_semantic.TensorboardSemanticClassVisualization(
                self.cfgs, save_as_tif=False, send_to_tensorboard=True
            ),
        ]
        if "corrupted" in self.cfgs.pipeline.semantic_dataset_type:
            viz += [
                visualize_semantic.TensorboardSemanticSummaryVisualization(
                    self.cfgs,
                    save_as_tif=False,
                    send_to_tensorboard=True,
                    compare_non_corrupted=True,
                )
            ]
        return viz

    @classmethod
    def init_config(cls, cfg_information):
        return RSSemanticConfig(**cfg_information)


class RSSemanticConfig(SatNeRFConfig):
    lambda_s: float = 0.04  # weight of the semantic loss. 0.04 in Semantic-NeRF Paper
    semantic_dataset_type: Literal["own", "us3d", "own_corrupted"] = "own"
    sparsity_n_images: int = -1  # use semantic labels for maximal n images
    ignore_car_index: Union[bool, int] = False
    # variations of semantic head/loss
    semantic_activation_function: Literal["none", "sigmoid"] = "sigmoid"
    use_tj_for_s: Union[bool, int] = False
    use_beta_for_s: Union[bool, int] = False
    use_tj_instead_of_beta: Union[bool, int] = False
    use_separate_beta_for_s: Union[bool, int] = False
    use_separate_tj_for_semantic: Union[bool, int] = False
    detach_beta_for_s: Union[bool, int] = False
    # car reg loss
    use_car_reg_loss: Union[bool, int] = False
    lambda_c: float = 1.0
    car_reg_loss_start: int = 3

    def determine_run_name_postfix(self):
        postfix = super().determine_run_name_postfix()
        if self.sparsity_n_images > 0:
            postfix += f"__sparsity{self.sparsity_n_images}"
        if self.ignore_car_index:
            postfix += "__ignorecars"
        if self.semantic_dataset_type != "own":
            if "corrupted" in self.semantic_dataset_type:
                postfix += "__corrupted"

        if self.use_tj_for_s:
            postfix += "__tj_for_s"
        if self.use_beta_for_s:
            if self.use_separate_beta_for_s:
                postfix += "__sbeta_for_s"
            else:
                postfix += "__cbeta_for_s"
            if self.detach_beta_for_s:
                postfix += "_detached"
        if self.use_tj_instead_of_beta:
            postfix += "__tj_instead_cbeta"
        if self.use_separate_tj_for_semantic:
            postfix += "__stj"

        if self.semantic_activation_function != "sigmoid":
            postfix += "__nsigm"

        if self.use_car_reg_loss:
            postfix += "__crl"
            if self.lambda_c != 1.0:
                postfix += str(self.lambda_c)

        return postfix
