import os
import torch
import gc
import time

from framework.pipelines import load_pipeline, run_pipeline
from framework.configs import load_configs
from framework.logger import logger
import framework.util.train_util as train_utils
from datetime import datetime


def start_training(run_config_fp, pipeline_config_fp):
    cfgs = load_configs(run_config_fp, pipeline_config_fp)
    start_training_cfgs(cfgs)


def start_training_cfgs(cfgs):
    create_run_dp_cfgs(cfgs)
    start_pipeline_cfgs(cfgs)


def start_pipeline_cfgs(cfgs):

    # deterministic
    if cfgs.run.deterministic:
        train_utils.reset_rng()
        torch.use_deterministic_algorithms(True)

    # create pipeline
    pipeline = load_pipeline(cfgs)
    # prepare run
    pipeline.prepare_run()
    # load the datasets
    pipeline.load_datasets()
    # start training
    run_pipeline(pipeline, cfgs)

    # for memory reasons when running automated trainings
    # delete the reference to the pipeline
    del pipeline


def create_run_dp_cfgs(cfgs):
    cfgs.create_run_name()
    os.makedirs(cfgs.run.run_dp, exist_ok=False)


def start_assigned_ids_from_automated(experiment_cfg_dp, gpu, *ids):
    for idx, id in enumerate(ids):
        if idx > 0:
            print("\n\n\n\n\n\n\n")
        logger.info("Automation", "=========================================")
        logger.info("Automation", f"Running experiment #{idx + 1}: '{id}' on cuda_{gpu}")
        logger.info("Automation", "=========================================")

        cfgs = load_configs(
            os.path.join(experiment_cfg_dp, f"{id}_run.toml"),
            os.path.join(experiment_cfg_dp, f"{id}_pipeline.toml"),
        )
        cfgs.run.gpu_id = gpu
        start_training_cfgs(cfgs)

        # clear the cache so the next training doesn't run out of memory
        torch.cuda.empty_cache()
        # make sure really everything that isn't needed anymore is gone
        gc.collect()
        # wait for some time just to make sure no weird CUDA errors occur
        time.sleep(60)


if __name__ == "__main__":
    import fire

    fire.Fire()
