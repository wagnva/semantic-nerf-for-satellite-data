import torch
from collections import defaultdict
import glob
import os
import numpy as np
import datetime
from tqdm import tqdm
import toml

from framework.logger import logger


@torch.no_grad()
def batched_inference(
    cfgs, renderer, models, rays, extras, render_options={}, epoch=None, show_tqdm=False
):
    """
    Do batched inference on rays using chunks.
    """
    chunk_size = cfgs.pipeline.render_chunk_size
    batch_size = rays.shape[0]
    results = defaultdict(list)

    steps = range(0, batch_size, chunk_size)
    if show_tqdm:
        steps = tqdm(steps)
    for i in steps:
        rendered_ray_chunks = renderer.render_rays(
            models,
            rays[i : i + chunk_size],
            extras[i : i + chunk_size] if extras is not None else None,
            epoch=epoch,
            render_options=render_options,
        )

        for k, v in rendered_ray_chunks.items():
            if k in results:
                results[k] = torch.cat([results[k], v], 0)
            else:
                results[k] = v

    return results


def expand_input_files_for_experiments(input_dp, output_dp=None):
    """
    Expands path to directory if pointing to an experiment folder
    Expands the output directory to contain the name of the experiment folder
        (even if input is to a single item of an experiment)
    :param input_dp: list of directory paths, either to an experiment or model
    :param output_dp: path to output, is expanded to contain the name of experiment folder if needed
    :return: expanded list of model_dps and the full output_dp with experiment included.
    """
    output = []

    # remove trailing slash as it leads to ensure consistent behaviour in the os.path methods
    input_dp = input_dp.rstrip("/")
    # check if given file is to a single model
    # is_single_training = len(glob.glob(os.path.join(input_dp, "ckpoints", "*.ckpt"))) > 0
    is_single_training = os.path.isdir(os.path.join(input_dp, "tensorboard"))
    if is_single_training:
        output.append(input_dp)

    else:
        sub_folders = [  # list of all immediate subdirectories
            os.path.join(input_dp, name)
            for name in os.listdir(input_dp)
            if os.path.isdir(os.path.join(input_dp, name))
        ]
        # filter sub_folders for dir not containing a model
        # by checking if they contain subfolder with the configs
        sub_folders = list(
            filter(
                lambda x: os.path.exists(os.path.join(x, "configs", "pipeline.toml")),
                sub_folders,
            )
        )
        output.extend(sub_folders)

    # sort the files
    output_files_sort_idx = np.argsort(output)
    output = np.array(output)[output_files_sort_idx]

    # extend the output directory with experiment category
    experiment_category = extract_experiment_category(output[0])
    if (
        experiment_category is not None
        and output_dp is not None
        and output_dp != input_dp
    ):
        # store the results in a dir with the same name as the experiment input folder
        output_dp = os.path.join(output_dp, experiment_category)
        os.makedirs(output_dp, exist_ok=True)

    return output, output_dp


def extract_experiment_category(training_dp):
    run_cfg_fp = os.path.join(training_dp, "configs", "run.toml")
    run_cfg = toml.load(run_cfg_fp)
    return "_" + run_cfg["experiment_category"]


def run_eval_script(
    run_eval_method,
    input_dp: str,
    output_dp: str = None,
    split="test",
    epoch=-1,
    skip_to_exp=1,
    device=0,
    device_req_free=True,
    **kwargs,
):
    """
    This method handles calling an evaluation script with all required parameters as specified in the command line.
    If no output_dp is given, it searches for an environment variable named 'SEMANTIC_SATNERF_EVAL_DP'.
    It extends the input_dp, if the path points to an experiment folder containing multiple trainings.
    The evaluation script is run separately for each of them.
    :param run_eval_method: the actual evaluation method
    :param input_dp: either a single training or an experiment folder containing multiple models
    :param output_dp: If no output_dp is given, the path specified in the environment variable 'SEMANTIC_SATNERF_EVAL_DP' is used instead
    :param split: 'train'|'split'
    :param epoch: if set, allows loading of a specific training epoch.
    :param skip_to_exp: if loading an experiment, this allows to skip until a certain experiment
    :param device: cuda device number. defaults to the first GPU (zero)
    :param device_req_free: if set to true, check if the GPU is free before starting the script to
    :param kwargs: additional arguments depending on the specified run_eval_method
    """
    if output_dp is None:
        output_dp = os.getenv("SEMANTIC_SATNERF_EVAL_DP")
        assert output_dp is not None and os.path.isdir(
            output_dp
        ), "No valid output dp specified as second argument. You can alternatively set a default output_dp in the ENV variable 'SEMANTIC_SATNERF_EVAL_DP'"
        logger.info(
            "Setup",
            f"Using eval output dp from ENV variable 'SEMANTIC_SATNERF_EVAL_DP' = {output_dp}",
        )
    else:
        assert os.path.exists(output_dp), "Invalid path given as output_dp"
        logger.info("Setup", f"Using specified eval output dp: {output_dp}")

    inputs, output_dp = expand_input_files_for_experiments(input_dp, output_dp)
    for input in inputs[(skip_to_exp - 1) :]:
        run_eval_method(
            input,
            output_dp=output_dp,
            split=split.lower(),
            epoch=epoch,
            device=device,
            device_req_free=device_req_free,
            **kwargs,
        )
