import torch

from framework.components.training_step import BaseTrainingStep

from baseline.components.training_step import SatNeRFTrainingStep

from semantic.components.metrics import semantic_accuracy, confusion_matrix


class RSSemanticTrainingStep(BaseTrainingStep):

    def training_step(self, pipeline, batch: torch.tensor, batch_idx: torch.tensor):
        rays = batch["rgb"]["rays"]  # (B, 11)
        rgbs = batch["rgb"]["rgbs"]  # (B, 3)
        extras = batch["rgb"]["extras"]  # (B, c_extras)

        results = pipeline(
            {"rays": rays, "extras": extras},
        )

        # RGB Loss
        if pipeline.get_current_epoch() < pipeline.cfgs.pipeline.first_beta_epoch:
            loss, loss_dict = pipeline.loss_without_beta(results, rgbs)
            pipeline.log("train/beta_loss_activated", 0.0)
        else:
            loss, loss_dict = pipeline.loss(results, rgbs)
            pipeline.log("train/beta_loss_activated", 1.0)

        # Depth Loss
        if pipeline.cfgs.pipeline.depth_enabled:
            if pipeline.train_steps < pipeline.ds_drop:

                tmp = pipeline(
                    {"rays": batch["depth"]["rays"], "extras": batch["depth"]["extras"]}
                )
                # logger.debug("Data", f"depth.rays.shape={batch['depth']['rays'].shape}")
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

        # Semantic Loss
        if (
            pipeline.get_current_epoch() < pipeline.cfgs.pipeline.first_beta_epoch
            or not pipeline.cfgs.pipeline.use_beta_for_s
        ):
            semantic_loss, semantic_loss_dict = pipeline.semantic_loss(
                results,
                batch["rgb"]["semantic"],
                batch["rgb"].get("semantic_sparsity_mask"),
            )
            pipeline.log("train/semantic_beta_loss_activated", 0.0)
        else:
            semantic_loss, semantic_loss_dict = pipeline.uncertainty_semantic_loss(
                results,
                batch["rgb"]["semantic"],
                batch["rgb"].get("semantic_sparsity_mask"),
            )
            pipeline.log("train/semantic_beta_loss_activated", 1.0)

        loss += semantic_loss
        for k in semantic_loss_dict.keys():
            loss_dict[k] = semantic_loss_dict[k]

        # Car Reg Loss
        if (
            pipeline.cfgs.pipeline.use_car_reg_loss
            and pipeline.get_current_epoch() >= pipeline.cfgs.pipeline.car_reg_loss_start
        ):
            car_reg_loss, car_reg_loss_dict = pipeline.car_reg_loss(
                results,
                batch["rgb"]["semantic"],
                batch["rgb"].get("semantic_sparsity_mask"),
            )
            loss += car_reg_loss
            for k in car_reg_loss_dict.keys():
                loss_dict[k] = car_reg_loss_dict[k]
            pipeline.log("train/car_reg_loss_activated", 1.0)

        pipeline.log(
            "train/semantic_accuracy",
            semantic_accuracy(results, batch["rgb"]["semantic"]),
        )

        return results, loss, loss_dict
