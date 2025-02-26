import abc
import importlib
import toml
import torch
import os
import shutil
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from lightning.pytorch.profilers import SimpleProfiler
import time

import framework.util.other
from framework.logger import logger as logger_text
import framework.util.train_util as train_utils
from framework.datasets import BaseDataset
from framework.components.rendering import BaseRenderer
from framework.components.training_step import BaseTrainingStep
from framework.util.train_util import create_cuda_device


class Pipeline(pl.LightningModule):
    def __init__(self, cfgs, ckpt_info=None) -> None:
        super().__init__()
        self.optimizer = None
        self.cfgs = cfgs
        self.workspace_dp = None
        self.logs_dp = None
        self.logger_text = logger_text
        self.train_steps = 0
        self.epoch_from_ckpt = None
        self.starting_new_training = ckpt_info is not None
        if ckpt_info is not None:
            self.epoch_from_ckpt, self.train_steps = ckpt_info

        self.datasets = self._init_datasets()
        assert (
            "rgb" in self.datasets.keys() and "rgb_test" in self.datasets.keys()
        ), "need both rgb and rgb_test datasets in pipeline"
        self.models = self.__init_models()
        self.renderer = self._init_renderer()
        self.visualizers = self._init_visualizers()
        self._training_step = self._init_training_step()
        self._time_of_last_step = None

        self._init_loss()

    def prepare_run(self):
        # create folder for the training
        self.workspace_dp = self.cfgs.run.run_dp
        # setup logger to print to log file inside of workspace dir
        self.logs_dp = os.path.join(self.workspace_dp, "configs")
        os.makedirs(self.logs_dp, exist_ok=False)
        self.logger_text.init_write_to_file(
            os.path.join(self.workspace_dp, "logger_text.log")
        )

        # copy over config files
        for key in ["run", "pipeline"]:
            with open(os.path.join(self.logs_dp, f"{key}.toml"), "wt") as fp:
                toml.dump(getattr(self.cfgs, key).dict(), fp)

        self.logger_text.info(
            "Setup",
            f"Preparing {self.cfgs.pipeline.pipeline} with run name: {self.cfgs.run.run_name}",
        )
        if (
            self.cfgs.run.experiment_category is not None
            and len(self.cfgs.run.experiment_category) > 0
        ):
            self.logger_text.info(
                "Setup",
                f"Storing inside of experiment folder: _{self.cfgs.run.experiment_category}",
            )

    def load_datasets(self):
        """
        Load all datasets.
        This method assumes that only rgb, rgb_test and depth datasets exist
        Needs adaption if new types of datasets are added later on.
        """
        assert {"rgb", "rgb_test"} <= set(
            self.datasets.keys()
        ), "Pipeline need both an rgb and rgb_test dataset"

        rgb_datasets = [self.datasets["rgb"], self.datasets["rgb_test"]]

        for idx, dataset in enumerate(rgb_datasets):
            self.logger_text.info(
                "Dataset",
                f"Loading dataset #{idx + 1}: {dataset.dataset_name}",
            )
            self.logger_text.subtopic()
            dataset.load()
            self.logger_text.info("Dataset", f"Dataset has size: {len(dataset)}")
            self.logger_text.reset_topic()

        self._handle_normalization()

    def configure_optimizers(self):
        """
        Lightning method: Configure optimizer and learnrate scheduler
        :return: dictionary containing optimizer and lr_scheduler
        """
        assert False, "needs to be implemented by subclass"

    def train_dataloader(self):
        d = {}
        for key in self.datasets:
            if not "test" in key:
                d[key] = DataLoader(
                    self.datasets[key],
                    shuffle=self.cfgs.run.shuffle_dataset,
                    num_workers=self.cfgs.run.train_n_workers,
                    batch_size=self.cfgs.pipeline.batch_size,
                    pin_memory=True,
                )
        return d

    def val_dataloader(self):
        for key in self.datasets:
            if "test" in key:
                return DataLoader(
                    self.datasets[key],
                    shuffle=False,
                    num_workers=self.cfgs.run.val_n_workers,
                    batch_size=1,  # validate one image (H*W rays) at a time
                    pin_memory=True,
                )

    def forward(self, data: dict):
        """
        Perform a single forward pass for the pipeline
        :param data: required data. Required fields: "rays" and "extras"
        :return: rendered results
        """
        assert False, "needs to be implemented by sub class"

    def training_step(self, batch, batch_idx):
        """
        Lightning method
        Handles a single training step
        :param batch:
        :param batch_idx:
        :return:
        """
        assert False, "needs to be implemented by sub class"

    def validation_step(self, batch, batch_idx):
        """
        Lightning method
        Handles a single validation step
        :param batch:
        :param batch_idx:
        :return:
        """
        assert False, "needs to be implemented by sub class"

    def get_current_epoch(self):
        if self.epoch_from_ckpt is not None:
            return self.epoch_from_ckpt
        return self.current_epoch

    def get_current_progress(self, tstep=None):
        if tstep is None:
            tstep = self.train_steps
        return tstep / self.cfgs.run.max_train_steps

    def get_progress_bar_dict(self):
        """
        Removes unnecessary v_num attribute on tqdm progress bar
        Just a small visual improvement
        :return: tqdm progress bar dict
        """
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def _handle_normalization(self):
        """
        Called when loading the datasets
        Handles interaction between datasets and normalization component
        """
        pass

    @abc.abstractmethod
    def _init_datasets(self) -> dict[BaseDataset]:
        """
        Init the datasets for training and validation.
        Has to be implemented by the respective pipeline
        :return:
        """
        pass

    @abc.abstractmethod
    def _init_loss(self):
        """
        Initialize the loss of the models
        Has to be implemented by the respective pipeline
        :return:
        """
        pass

    def __init_models(self) -> dict:
        """
        Adapt the models in the dict returned by the subclass
        Pytorch Lightning expects each of the models to be a class variable
        :return: dict containing the different models
        """
        models = self._init_models()
        for key in models.keys():
            setattr(self, f"model_{key}", models[key])
            models[key] = getattr(self, f"model_{key}")
        return models

    @abc.abstractmethod
    def _init_models(self) -> dict:
        """
        Initialize the models
        Has to be implemented by the respective pipeline
        :return: dict containing the different models
        """
        pass

    @abc.abstractmethod
    def _init_renderer(self):
        pass

    @abc.abstractmethod
    def _init_visualizers(self) -> list:
        pass

    @abc.abstractmethod
    def _init_training_step(self):
        pass


def run_pipeline(pipeline, cfgs):
    """
    Run a given training pipeline using Pytorch Lightning
    :param pipeline: the instantiated pipeline
    :param cfg: config
    :return: Nothing
    """
    torch.cuda.empty_cache()

    # cuda_device = create_cuda_device(device=cfgs.run.gpu_id, device_req_free=cfgs.run.device_req_free)
    # pipeline.to(cuda_device)

    # set the max memory
    torch.cuda.set_per_process_memory_fraction(
        cfgs.run.gpu_max_memory_fraction, cfgs.run.first_free_gpu()
    )
    torch.set_float32_matmul_precision(
        cfgs.run.float32_matmul_precision
    )  # 'medium' | 'high'

    # save the one with the best psnr
    # does not work when resuming from .ckpt, needs to be changed
    ckpt_callback_psnr = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfgs.run.run_dp, "ckpoints"),
        filename="{epoch:d}_best_psnr",
        monitor="train/psnr",
        mode="max",
        every_n_epochs=1,
    )
    # save the one with the lowest mae
    ckpt_callback_mae = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfgs.run.run_dp, "ckpoints"),
        filename="{epoch:d}_best_mae",
        monitor="train/mae",
        mode="min",
        every_n_epochs=1,
    )
    callbacks = [ckpt_callback_mae]  # ckpt_callback_psnr , ckpt_callback_mae]
    if cfgs.run.save_every_n_epochs is not None and cfgs.run.save_every_n_epochs > 0:
        # This stores every n epoch .ckpt as configured
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(cfgs.run.run_dp, "ckpoints"),
            filename="{epoch:d}",
            every_n_epochs=cfgs.run.save_every_n_epochs,
            save_last=True,
            save_top_k=-1,
        )
    else:
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(cfgs.run.run_dp, "ckpoints"),
            filename="{epoch:d}",
            every_n_epochs=0,
            save_last=True,
            save_top_k=0,
        )
    callbacks.append(ckpt_callback)

    logger = TensorBoardLogger(
        save_dir=cfgs.run.run_dp,
        name="",  # the save dir is already the complete path, so name is set to empty
        default_hp_metric=False,
        version="tensorboard",  # sets the folder name for the logs
    )

    profiler = SimpleProfiler(
        dirpath=cfgs.run.run_dp,
        filename="profiler",
    )
    trainer = pl.Trainer(
        max_steps=cfgs.run.max_train_steps,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=[cfgs.run.gpu_id],
        # Setting this to true leads to error -> Setting torch.use_deterministic_algorithms(True) works
        deterministic=False,
        benchmark=True,
        # weights_summary=None,
        num_sanity_val_steps=cfgs.run.num_sanity_val_steps,
        check_val_every_n_epoch=cfgs.run.check_val_every_n_epoch,
        profiler=profiler,
        precision=cfgs.pipeline.precision,
    )

    ckpoint_fp = None
    if cfgs.run.resume_from_ckpoint and cfgs.run.ckpoint_fp is not None:
        assert os.path.isfile(cfgs.run.ckpoint_fp), ".ckpt file not found"
        ckpoint_fp = cfgs.run.ckpoint_fp
        logger_text.info("Pipeline", f"Continuing training from ckpoint: {ckpoint_fp}")

    time_start = time.time()

    # train the model
    trainer.fit(pipeline, ckpt_path=ckpoint_fp)

    time_diff = time.time() - time_start

    logger_text.info(
        "Pipeline",
        f"Finished training after {pipeline.get_current_epoch()} epochs in {time_diff:.2f}s = {(time_diff/60):.2f}min = {(time_diff/60/60):.2f}h",
    )


def load_pipeline(cfgs, ckpt_info=None) -> Pipeline:
    """
    Load and instantiate a pipeline based on the configs given
    :param cfgs: dict containing 'pipeline' and 'location' config
    :return: The instantiated pipeline class
    """
    # load and create the pipeline class specified in the config
    name = cfgs.pipeline.pipeline.split(".")
    pipeline_file = importlib.import_module(".".join(name[:-1]))
    pipeline = getattr(pipeline_file, name[-1])(cfgs, ckpt_info=ckpt_info)

    return pipeline
