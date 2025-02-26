import os
import glob
import torch
import gpustat
from pytorch_lightning import Trainer, LightningModule

from framework.logger import logger
from framework.pipelines import load_pipeline
from framework.util.train_util import create_cuda_device


def find_ckpoint_fp(log_dp, epoch=-1):
    if epoch >= 0:
        checkpoint_fp = os.path.join(log_dp, "ckpoints", f"epoch={epoch}.ckpt")
    else:
        checkpoint_fp = os.path.join(log_dp, "ckpoints", f"last.ckpt")
        if not os.path.isfile(checkpoint_fp):
            # path to the last epoch
            checkpoint_fp = sorted(
                glob.glob(os.path.join(log_dp, "ckpoints", f"*.ckpt")),
                key=lambda x: int(x[x.index("=") + 1 : x.index(".ckpt")]),
            )
            assert len(checkpoint_fp) > 0, "cannot find a single *.ckpt to load"
            checkpoint_fp = checkpoint_fp[-1]
            x = os.path.basename(checkpoint_fp)
            epoch = int(x[x.index("=") + 1 : x.index(".ckpt")])
    return checkpoint_fp, epoch


def load_from_disk(
    cfgs, log_dp: str, epoch=-1, device=0, device_req_free=True, prefixes_to_ignore=[]
):
    """
    Loads models from disk, based on a checkpoint
    :param cfgs: loaded configs
    :param log_dp: path to logs directory of model
    :param epoch: which epoch to load. If -1, then load the latest one
    :param device: load model on which cuda device
    :param device_req_free: if set to true, the device is required to be free with no other process using it
    :return: the instantiated models, with loaded weights, and the associated pipeline
    """
    if epoch is not None:
        checkpoint_fp, epoch = find_ckpoint_fp(log_dp, epoch=epoch)
        logger.info("Setup", f"Loading from ckpoint: {checkpoint_fp}")

        if not os.path.exists(checkpoint_fp):
            raise FileNotFoundError("Could not find checkpoint {}".format(checkpoint_fp))
    else:
        checkpoint_fp = None
        logger.info("Setup", f"Creating Pipeline without loading weights from ckpoint")

    device = create_cuda_device(device, device_req_free)

    logger.info("Setup", f"Using device: '{device}'")

    # load epoch number from .ckpt file
    # done manually, since the way we initialize the pipeline it is not loaded
    ckpt = torch.load(checkpoint_fp)
    ckpt_info = (ckpt["epoch"], ckpt["global_step"])

    # create the pipeline, so we can extract the models
    pipeline = load_pipeline(cfgs, ckpt_info=ckpt_info)
    pipeline = pipeline.to(device)
    models = pipeline.models

    # load from ckpoint for each model
    if epoch is not None:
        for key in models:
            model_name = f"model_{key}"  # the way the models are named as attributes in the pipeline
            load_ckpoint(
                models[key], checkpoint_fp, model_name, device, prefixes_to_ignore
            )
            models[key] = models[key].eval()

    return models, pipeline, pipeline.get_current_epoch(), device


def load_ckpoint(model, checkpoint_fp, model_name, cuda_device, prefixes_to_ignore=[]):
    """
    Load the weights for a given model from a ckpt file
    :param model: the model
    :param checkpoint_fp: path to the checkpoint file
    :param model_name: name of the model, if the ckpt file contains info for multiple models
    :param cuda_device: the cuda device to load the models on
    """
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(
        checkpoint_fp, model_name, cuda_device, prefixes_to_ignore
    )
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def extract_model_state_dict(
    checkpoint_fp, model_name, cuda_device, prefixes_to_ignore=[], prefixes_to_load=None
):
    """
    For a given checkpoint, load only the weights related to a certain model
    :param checkpoint_fp: path to checkpoint
    :param model_name: load only weights for this model based on name prefix
    :param prefixes_to_ignore: ignore a specific prefix
    :return: weights for a model
    """
    checkpoint = torch.load(checkpoint_fp, map_location=cuda_device, weights_only=True)
    checkpoint_ = {}
    if "state_dict" in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint["state_dict"]
    for k, v in checkpoint.items():
        if not k.startswith(model_name + "."):
            continue
        k = k[len(model_name) + 1 :]

        ignore = prefixes_to_load is not None
        # handle ignore prefixes
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                ignore = True
                break

        # handle only loading required prefixes
        if prefixes_to_load is not None:
            for prefix in prefixes_to_load:
                if k.startswith(prefix):
                    ignore = False
                    break

        if not ignore:
            checkpoint_[k] = v
    return checkpoint_
