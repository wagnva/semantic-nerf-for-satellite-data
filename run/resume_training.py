import os

from framework.configs import load_configs_from_logs
from framework.util.load_ckpoint import find_ckpoint_fp
from framework.logger import logger
from run.training import start_training_cfgs


"""
This script makes it faster to continue an training.
Instead of having to set the corresponding path in the configs,
this script takes the path to an existing training and restarts/continues it based on the local configuration.
"""


def resume_training(log_dp: str, epoch=-1):
    assert os.path.isdir(log_dp), "log_dp is not a path to an existing folder"

    cfgs = load_configs_from_logs(log_dp)

    ckpoint_fp, epoch = find_ckpoint_fp(log_dp, epoch=epoch)
    assert os.path.isfile(ckpoint_fp), "cannot find an existing ckpoint"

    logger.info("Resume", f"Resuming from: {os.path.basename(ckpoint_fp)}")

    # update the path in the config so that the training knows where to resume the training from
    cfgs.run.ckpoint_fp = ckpoint_fp
    cfgs.run.resume_from_ckpoint = True

    start_training_cfgs(cfgs)


if __name__ == "__main__":
    import fire

    fire.Fire(resume_training)
