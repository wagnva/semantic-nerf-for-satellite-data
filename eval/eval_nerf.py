import glob
import os
from tqdm import tqdm
import glob
import numpy as np
import json

from eval.utils.util import batched_inference, run_eval_script
from framework.configs import load_configs_from_logs, adapt_configs_for_inference
from framework.util.load_ckpoint import load_from_disk
import eval.utils.dsm as dsm_util
import eval.utils.metrics as metrics
from framework.util.other import w_h_from_sample


def eval_nerf_training(
    input_dp: str,
    output_dp: str = None,
    split="test",
    epoch=-1,
    device=0,
    device_req_free=True,
):
    assert os.path.isdir(input_dp), "input_dp is not a path to an existing folder"

    cfgs = load_configs_from_logs(input_dp)
    cfgs = adapt_configs_for_inference(cfgs)

    output_dp = os.path.join(output_dp, cfgs.run.run_name, "eval", split)
    os.makedirs(output_dp, exist_ok=True)

    # load trained nerf
    models, pipeline, epoch, cuda_device = load_from_disk(
        cfgs, input_dp, epoch, device, device_req_free
    )
    # load the datasets
    pipeline.load_datasets()

    dataset_name = "rgb"
    if split == "test":
        dataset_name = "rgb_test"

    dataset = pipeline.datasets[dataset_name]
    dataset.force_act_as_test()

    all = {}

    start = 0
    if split == "test":
        # ignore the first item, since it is also a training view
        start = 1

    progress_bar = tqdm(range(start, len(dataset)))
    progress_bar.set_description(f"MAE (Mean) = {0.0:.3f}, MAE (Median) = {0.0:.3f}")

    for img_idx in progress_bar:
        img = dataset[img_idx]
        rays = img["rays"].to(cuda_device)
        extras = img["extras"].to(cuda_device)

        results = batched_inference(cfgs, pipeline.renderer, models, rays, extras)

        typ = "fine" if "rgb_fine" in results else "coarse"
        depths = results[f"depth_{typ}"]

        rays = rays.cpu()
        depths = depths.cpu()
        rgbs = img["rgbs"].squeeze().to(cuda_device)

        mae = dsm_util.compute_dsm_and_mae(
            dataset, rays, depths, output_dp, img["name"], epoch
        )

        # print(mae)
        # calculate psnr and ssim
        W, H = w_h_from_sample(img)
        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(
            results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.reshape(1, 3, H, W)
        )

        all[img["name"]] = {
            "mae": mae,
            "psnr": "{:.2f}".format(psnr_),
            "ssim": "{:.3f}".format(ssim_),
        }

        # calculate the average values over the metrics
        complete_mae_mean, complete_mae_median = 0.0, 0.0
        psnr_mean, ssim_mean = 0.0, 0.0
        for value in all.values():
            complete_mae_mean += float(value["mae"]["mean"])
            complete_mae_median += float(value["mae"]["median"])
            psnr_mean += float(value["psnr"])
            ssim_mean += float(value["ssim"])

        complete_mae_mean /= len(all)
        complete_mae_median /= len(all)
        psnr_mean /= len(all)
        ssim_mean /= len(all)

        progress_bar.set_description(
            f"MAE (Mean) = {complete_mae_mean:.3f}, MAE (Median) = {complete_mae_median:.3f}, PSNR = {psnr_mean:.2f}, SSIM = {ssim_mean:.3f}"
        )

        stats_fp = os.path.join(output_dp, "results.json")
        with open(stats_fp, "w") as f:
            d = all.copy()
            d["MAE (Mean)"] = "{:.3f}".format(complete_mae_mean)
            d["MAE (Median)"] = "{:.3f}".format(complete_mae_median)
            d["PSNR (Mean)"] = "{:.2f}".format(psnr_mean)
            d["SSIM (Mean)"] = "{:.3f}".format(ssim_mean)
            json.dump(d, f, indent=4)

    print("Complete Evaluation Results")
    print(json.dumps(d, indent=4))


if __name__ == "__main__":
    import fire

    fire.Fire(
        lambda *args, **kwargs: run_eval_script(eval_nerf_training, *args, **kwargs)
    )
