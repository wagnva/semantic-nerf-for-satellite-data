import torch
from collections import defaultdict
import os
import time

from framework.util.other import w_h_from_sample
from framework.pipelines import Pipeline
import framework.util.train_util as train_utils

import eval.utils.metrics as metrics
import eval.utils.dsm as dsm


class BaseRayPipeline(Pipeline):

    def forward(self, data: dict, render_options: dict = defaultdict(bool)):
        """
        Perform a single forward pass for the pipeline
        :param data: required data. Required fields: "rays" and "extras"
        :return: rendered results
        """
        rays = data["rays"]
        extras = data["extras"]
        epoch = data.get("epoch", self.get_current_epoch())
        progress = data.get("progress", self.get_current_progress())
        chunk_size = self.cfgs.pipeline.render_chunk_size
        batch_size = rays.shape[0]
        results = defaultdict(list)

        # render_options.update({
        #     "inference": not self.training
        # })

        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = self.renderer.render_rays(
                self.models,
                rays[i : i + chunk_size],
                extras[i : i + chunk_size] if extras is not None else None,
                epoch=epoch,
                progress=progress,
                render_options=render_options,
            )

            for k, v in rendered_ray_chunks.items():
                if k in results:
                    if v is not None:
                        if results[k] is None:
                            results[k] = v
                        else:
                            results[k] = torch.cat([results[k], v], 0)
                else:
                    results[k] = v

        return results

    def training_step(self, batch, batch_idx):
        """
        Lightning method
        Handles a single training step
        :param batch:
        :param batch_idx:
        :return:
        """

        self.log("lr", train_utils.get_learning_rate(self.optimizer))
        self.train_steps += 1
        batch_size = batch["rgb"]["rays"].shape[0]

        results, loss, loss_dict = self._training_step.training_step(
            self, batch, batch_idx
        )

        # self.cfgs["state"]["noise_std"] *= 0.9

        self.log("train/loss", loss, batch_size=batch_size)
        typ = "fine" if "rgb_fine" in results else "coarse"

        with torch.no_grad():
            psnr_ = metrics.psnr(results[f"rgb_{typ}"], batch["rgb"]["rgbs"])
        for k in loss_dict.keys():
            self.log("train/{}".format(k), loss_dict[k], batch_size=batch_size)

        self.log(
            "train/psnr",
            psnr_,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if self._time_of_last_step is None:
            self._time_of_last_step = time.time()
        time_diff = time.time() - self._time_of_last_step
        self.log("train/time_since_last_step", time_diff)
        self.log("train/time_it_p_sec", 1.0 / time_diff)
        self._time_of_last_step = time.time()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Lightning method
        Handles a single validation step
        :param batch:
        :param batch_idx:
        :return:
        """
        split = batch["split"][0]
        rays, rgbs, extras = batch["rays"], batch["rgbs"], batch["extras"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        extras = extras.squeeze()  # (H*W, c_extras)

        # self.logger_text.info(
        #    "Validation", f"Validating image: {batch['name'][0]} ({split})"
        # )

        assert (
            rays.shape[0] == rgbs.shape[0]
        ), "Rays&RGBs shape dont match (validation step)"

        epoch = self.get_current_epoch()
        data = {"rays": rays, "extras": extras, "epoch": epoch}
        render_options = self._val_render_options(split)
        render_options["split"] = split
        with torch.no_grad():
            results = self(data, render_options)

        loss, loss_dict = self.loss(results, rgbs)

        W, H = w_h_from_sample(batch)

        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(
            results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W)
        )

        # need to subtract one from index for test images, since the training images is the first image
        sample_idx = batch_idx - 1 if split == "test" else batch_idx

        display_epoch = self.get_current_epoch()
        if self.train_steps > 0:
            # display the epoch number as how many epochs are finished (i.e. epoch 1 => one epoch has been trained)
            # validation sanity steps should be displayed as epoch 0 (therefore train_steps > 0)
            display_epoch += 1

        for visualizer in self.visualizers:
            visualizer.run(
                self,
                self.datasets["rgb_test"],
                batch,
                results,
                sample_idx=sample_idx,
                split=split,
                epoch=display_epoch,
                source_fp=batch["img_fp"][0],
                logger=self.logger,
            )

        # only have the actual validation images contribute to the metrics
        if split == "test":
            self.log("test/loss", loss, batch_size=1)
            self.log("test/psnr", psnr_, batch_size=1)

        # ssim is logged for both splits
        self.log(f"{split}/ssim", ssim_, batch_size=1)

        # compute mae for the single training image and first test image
        if batch_idx <= 1:
            output_dp = os.path.join(
                self.cfgs.run.run_dp,
                "visualization",
                split,
                "dsm",
            )
            mae = dsm.compute_dsm_and_mae(
                self.datasets["rgb_test"],
                rays.cpu(),
                results[f"depth_{typ}"].cpu(),
                output_dp,
                batch["name"][0],
                self.get_current_epoch(),
            )
            self.log(f"{split}/mae", float(mae["mean"]), batch_size=1)
            # self.logger.experiment.add_scalars("train/mae", {'MAE (Mean)': mae['mean']}, self.global_step)
            # self.logger.experiment.add_scalars("train/mae", {'MAE (Median)': mae['median']}, self.global_step)

        # recombine dataset
        # this is needed when using epoch sub sampling
        for dataset in self.datasets.values():
            dataset.recombine_if_needed()

    def _val_render_options(self, split):
        return {}

    def _handle_normalization(self):
        # handle normalization
        self.logger_text.info("Dataset", "Normalize rays from rgb and rgb_test datasets")
        self.logger_text.subtopic()
        rgb_combined_rays = torch.cat(
            (
                self.datasets["rgb"].combined_data["rays"],
                self.datasets["rgb_test"].combined_data["rays"],
            ),
            dim=0,
        )
        rgb_datasets = [self.datasets["rgb"], self.datasets["rgb_test"]]

        # initialize normalization component for rgb datasets
        for idx, dataset in enumerate(rgb_datasets):
            dataset.initialize_normalization(combined_data={"rays": rgb_combined_rays})

        # normalize rgb datasets
        for idx, dataset in enumerate(rgb_datasets):
            self.logger_text.info(
                "Dataset",
                f"Normalize dataset #{idx + 1} of {len(self.datasets)}: {dataset.dataset_name}",
            )

            # save rays to cache if needed
            # save the unnormalized rays
            dataset.save_to_cache()
            dataset.normalize()

        self.logger_text.reset_topic()

        # handle depth dataset if existing
        # handled this way at the end, since it depends on normalization of rgb datasets during its dataset creation
        if "depth" in self.datasets.keys():
            dataset = self.datasets["depth"]

            self.logger_text.info(
                "Dataset",
                f"Loading dataset #3: {dataset.dataset_name}",
            )
            self.logger_text.subtopic()
            # initialize normalization component before loading the dataset
            dataset.initialize_normalization()
            dataset.load()
            dataset.normalize()
            self.logger_text.info("Dataset", f"Dataset has size: {len(dataset)}")
            self.logger_text.reset_topic()

    def configure_optimizers(self):
        """
        Lightning method: Configure optimizer and learnrate scheduler
        :return: dictionary containing optimizer and lr_scheduler
        """
        parameters = train_utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(
            parameters, lr=self.cfgs.pipeline.learnrate, weight_decay=0
        )

        max_epochs = train_utils.get_epoch_number_from_train_step(
            self.cfgs.run.max_train_steps,
            len(self.datasets["rgb"]),
            self.cfgs.pipeline.batch_size,
        )
        scheduler = train_utils.get_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="step",
            num_epochs=max_epochs,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
