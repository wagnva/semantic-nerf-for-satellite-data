from framework.components.rendering import BaseRenderer
from framework.pipelines import Pipeline
from framework.visualize import BaseVisualization
from framework.components.training_step import BaseTrainingStep
from baseline.components.training_step import NeRFTrainingStep
from baseline.dataset.satnerf_dataset import SatNeRFDataset
from baseline.models.snerf import ShadowNeRF
from baseline.components.rendering import SNeRFRendering
from baseline.components.loss import SNerfLoss
import baseline.components.visualize as visualize
from baseline.pipelines.nerf import NeRFConfig


class SNerfPipeline(Pipeline):
    def _init_datasets(self) -> dict:
        return {
            "rgb": SatNeRFDataset(self.cfgs, "rgb", "train"),
            "rgb_test": SatNeRFDataset(self.cfgs, "rgb", "test"),
        }

    def _init_loss(self):
        self.loss = SNerfLoss(lambda_sc=self.cfgs.pipeline.sc_lambda)

    def _init_models(self) -> dict:
        d = {
            "coarse": ShadowNeRF(
                layers=self.cfgs.pipeline.fc_layers,
                feat=self.cfgs.pipeline.fc_units,
                skips=self.cfgs.pipeline.fc_skips,
            )
        }
        return d

    def _init_renderer(self) -> BaseRenderer:
        return SNeRFRendering(self.cfgs)

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
            visualize.FactorVisualization(
                self.cfgs,
                save_as_tif=True,
                send_to_tensorboard=True,
                factor_name="albedo",
            ),
            visualize.FactorVisualization(
                self.cfgs, save_as_tif=True, send_to_tensorboard=True, factor_name="sun"
            ),
        ]

    @classmethod
    def init_config(cls, cfg_information):
        return SNeRFConfig(**cfg_information)


class SNeRFConfig(NeRFConfig):
    sc_lambda: float = 0.05
