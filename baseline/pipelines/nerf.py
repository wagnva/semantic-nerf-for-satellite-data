from pydantic import BaseModel
from typing import List, Union, Literal

from framework.components.rendering import BaseRenderer
from framework.components.training_step import BaseTrainingStep
from framework.pipelines import Pipeline
from framework.visualize import BaseVisualization
from baseline.dataset.satnerf_dataset import SatNeRFDataset
from baseline.models.nerf import NeRF
from baseline.components.rendering import NeRFRendering
from baseline.components.loss import NerfLoss
from baseline.components.training_step import NeRFTrainingStep
import baseline.components.visualize as visualize


class NerfPipeline(Pipeline):
    def _init_datasets(self) -> dict:
        return {
            "rgb": SatNeRFDataset(self.cfgs, "rgb", "train"),
            "rgb_test": SatNeRFDataset(self.cfgs, "rgb", "test"),
        }

    def _init_loss(self):
        self.loss = NerfLoss()

    def _init_models(self) -> dict:
        d = {
            "coarse": NeRF(
                layers=self.cfgs.pipeline.fc_layers,
                feat=self.cfgs.pipeline.fc_units,
                skips=self.cfgs.pipeline.fc_skips,
            )
        }
        return d

    def _init_renderer(self) -> BaseRenderer:
        return NeRFRendering(self.cfgs)

    def _init_training_step(self) -> BaseTrainingStep:
        return NeRFTrainingStep()

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
            visualize.RGBDiffDistanceVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True
            ),
        ]

    @classmethod
    def init_config(cls, cfg_information):
        return NeRFConfig(**cfg_information)


class NeRFConfig(BaseModel):
    pipeline: str = None
    precision: int = 32
    use_utm_coordinate_system: Union[bool, int] = False
    version: int = 1

    n_samples: int = 64
    render_chunk_size: int = 5120
    batch_size: int = 1024
    learnrate: float = 5e-4
    noise_std: float = 0.0
    activation_function: Literal["siren", "relu"] = "siren"
    mapping_pos_n_freq: int = 10  # number of frequencies used for positional mapping
    mapping_dir_n_freq: int = 4  # number of frequencies used for direction mapping

    fc_units: int = 512
    fc_layers: int = 8
    fc_skips: list[int] = [4]

    ray_subsampling_activated: Union[bool, int] = False
    ray_subsampling_amount: float = 1.0

    def determine_run_name_postfix(self):
        if self.use_utm_coordinate_system:
            return "utm"
        return ""
