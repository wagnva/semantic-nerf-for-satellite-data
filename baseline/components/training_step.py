import torch

from framework.components.training_step import BaseTrainingStep
from framework.logger import logger


class NeRFTrainingStep(BaseTrainingStep):
    def training_step(self, pipeline, batch: torch.tensor, batch_idx: torch.tensor):
        rays = batch["rgb"]["rays"]  # (B, 11)
        rgbs = batch["rgb"]["rgbs"]  # (B, 3)
        extras = batch["rgb"]["extras"]  # (B, c_extras)

        results = pipeline({"rays": rays, "extras": extras})

        loss, loss_dict = pipeline.loss(results, rgbs)
        return results, loss, loss_dict


class SatNeRFTrainingStep(BaseTrainingStep):
    def training_step(self, pipeline, batch: torch.tensor, batch_idx: torch.tensor):
        rays = batch["rgb"]["rays"]  # (B, 11)
        rgbs = batch["rgb"]["rgbs"]  # (B, 3)
        extras = batch["rgb"]["extras"]  # (B, c_extras)

        results = pipeline({"rays": rays, "extras": extras})

        if pipeline.get_current_epoch() < pipeline.cfgs.pipeline.first_beta_epoch:
            loss, loss_dict = pipeline.loss_without_beta(results, rgbs)
            pipeline.log("train/beta_loss_activated", 0.0)
        else:
            loss, loss_dict = pipeline.loss(results, rgbs)
            pipeline.log("train/beta_loss_activated", 1.0)

        if pipeline.cfgs.pipeline.depth_enabled:
            if pipeline.train_steps < pipeline.ds_drop:

                tmp = pipeline(
                    {"rays": batch["depth"]["rays"], "extras": batch["depth"]["extras"]}
                )
                kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])

                kp_weights = (
                    1.0
                    if pipeline.cfgs.pipeline.ds_noweights
                    else torch.flatten(batch["depth"]["weights"])
                )
                loss_depth, loss_dict_depth = pipeline.depth_loss(
                    tmp, kp_depths, kp_weights
                )

                loss += loss_depth
                for k in loss_dict_depth.keys():
                    loss_dict[k] = loss_dict_depth[k]

                pipeline.log("train/depth_loss_activated", 1.0)
            else:
                pipeline.log("train/depth_loss_activated", 0.0)

        return results, loss, loss_dict
