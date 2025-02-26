import numpy as np
import os
import pandas as pd
from pyntcloud import PyntCloud
from tqdm import tqdm
import torch

from framework.logger import logger
from framework.configs import load_configs_from_logs, adapt_configs_for_inference
from framework.util.load_ckpoint import load_from_disk
from framework.components.rays import extras_component_fn
from eval.utils.util import batched_inference, run_eval_script
import eval.utils.dsm as dsm_util
from framework.util.train_util import reset_rng


import warnings

warnings.filterwarnings("ignore")


def extract_sun_dirs(rays, extras, results):
    return extras_component_fn(extras.cpu(), "sun_d")


def create_dsm_pointcloud(
    log_dp: str,
    output_dp: str,
    split="test",
    epoch=-1,
    device=0,
    device_req_free=True,
    results_dir_name="pointclouds",
    max_items=1000000,
    normals_fn=extract_sun_dirs,
    render_options_fn=lambda: {},
    save_fns=[],
):
    assert os.path.isdir(log_dp), "log_dp is not a path to an existing folder"

    cfgs = load_configs_from_logs(log_dp)
    cfgs = adapt_configs_for_inference(cfgs)

    output_dp = os.path.join(output_dp, cfgs.run.run_name, results_dir_name, split)
    os.makedirs(output_dp, exist_ok=True)

    # load trained nerf
    models, pipeline, epoch, cuda_device = load_from_disk(
        cfgs, log_dp, epoch, device, device_req_free
    )

    # load the datasets
    pipeline.load_datasets()

    dataset_name = "rgb"
    if split == "test":
        dataset_name = "rgb_test"

    dataset = pipeline.datasets[dataset_name]
    dataset.force_act_as_test()

    until = min(max_items, len(dataset))

    save_fns += [save_ply]  # always store pcs as .ply

    for img_idx in tqdm(range(until)):
        img = dataset[img_idx]
        rays = img["rays"].to(cuda_device)
        extras = img["extras"].to(cuda_device)

        results = batched_inference(
            cfgs,
            pipeline.renderer,
            models,
            rays,
            extras,
            render_options=render_options_fn(),
        )

        typ = "fine" if "rgb_fine" in results else "coarse"
        depth = results[f"depth_{typ}"]

        rays = rays.cpu()
        depth = depth.cpu()

        dsm_n = dataset.get_xyz_from_nerf_prediction(rays, depth)
        dsm = dsm_util.create_dsm_cloud_from_nerf(dataset, rays, depth)
        colors = results[f"rgb_{typ}"].cpu().numpy()
        normals = normals_fn(rays, extras, results)

        name = f"{img['name']}_epoch_{epoch}"

        save_output(save_fns, dsm, colors, normals, output_dp, name)
        save_output(save_fns, dsm_n, colors, normals, output_dp, name, "normalized")

        # create a filtered version of the pointclouds with fewer points
        # for comparison purposes, reset rng
        # to make sure filtered pointcloud always has the same selection of points
        reset_rng()
        indices = torch.randperm(len(dsm_n))[:30000]

        save_output(save_fns, dsm, colors, normals, output_dp, name, indices=indices)
        save_output(
            save_fns,
            dsm_n,
            colors,
            normals,
            output_dp,
            name,
            "normalized",
            indices=indices,
        )

    logger.info("Cloud", f"Extracted {until} pointclouds to {output_dp}")


def save_output(
    save_fns, xyz, colors, normals, output_dp, name, postfix="", indices=None
):

    if len(postfix) > 0:
        name += f"_{postfix}"

    if indices is not None:
        xyz = xyz[indices]
        colors = colors[indices]
        normals = normals[indices]
        name += "_filtered"

    output_fp = os.path.join(output_dp, name)

    for save_fn in save_fns:
        save_fn(xyz, colors, normals, output_fp)


def save_ply(points, colors, normals, output_fp):
    # convert to pandas dataframe
    points = np.concatenate((points, colors, normals), axis=-1)
    points = pd.DataFrame(
        points, columns=["x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"]
    )

    # create pyntcloud
    cloud = PyntCloud(points)
    cloud.to_file(output_fp + ".ply")


if __name__ == "__main__":
    import fire

    fire.Fire(
        lambda *args, **kwargs: run_eval_script(
            create_dsm_pointcloud,
            *args,
            **kwargs,
        )
    )
