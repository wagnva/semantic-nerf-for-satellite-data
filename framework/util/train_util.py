from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    StepLR,
)
import pytorch_lightning as pl
import time
import torch
import gpustat
import random
import numpy as np


def get_epoch_number_from_train_step(train_step, dataset_len, batch_size):
    return int(train_step // (dataset_len // batch_size))


def get_learning_rate(optimizer):
    """
    Get learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_parameters(models):
    """
    Get all model parameters recursively
    models can be a list, a dictionary or a single pytorch model
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_scheduler(optimizer, lr_scheduler, num_epochs):
    eps = 1e-8
    if lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eps)
    elif lr_scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.01)
    elif lr_scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
        # scheduler = MultiStepLR(optimizer, milestones=[50,100,200], gamma=0.5)
    elif lr_scheduler == "step":
        gamma = 0.9
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    else:
        raise ValueError("lr scheduler not recognized!")

    return scheduler


def create_cuda_device(device=0, device_req_free=True, memory_req_free_ratio=0.05):
    """
    Creates a cuda device, making sure the request device is not already utilized by another process
    """
    assert check_cuda_device_usable(device, device_req_free, memory_req_free_ratio), (
        "Trying to use a GPU that is already utilized by another process. Set --device_req_free=False to disable "
        "this assert"
    )
    return torch.device(f"cuda:{device}")


def check_cuda_device_usable(device=0, device_req_free=True, memory_req_free_ratio=0.05):
    assert device < torch.cuda.device_count(), "Trying to use a GPU that doesn't exist"
    if device_req_free:
        stats = gpustat.GPUStatCollection.new_query()
        assert torch.cuda.device_count() == len(
            stats
        ), "Error trying to query information about the available GPUs"
        # check if at least 95% percent of memory is free
        # --> should be available and not used by any other relevant process
        gpu_stat = stats[device]
        mem_util = float(gpu_stat.entry["memory.used"]) / float(
            gpu_stat.entry["memory.total"]
        )
        return mem_util <= memory_req_free_ratio

    return True


def get_list_of_free_cuda_devices(allowed=None):
    devices = range(torch.cuda.device_count())
    free_devices = list(
        filter(lambda x: check_cuda_device_usable(x, device_req_free=True), devices)
    )

    if allowed is not None:
        allowed = allowed if isinstance(allowed, list) else [allowed]
        filtered_devices = list(filter(lambda x: x in allowed, free_devices))
        assert (
            len(filtered_devices) > 0
        ), f"None of the configured GPU(s) are free. Configured = {allowed}, Free = {free_devices}"
        free_devices = filtered_devices
    return free_devices


def reset_rng(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
