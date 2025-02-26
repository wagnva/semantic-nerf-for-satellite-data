import glob
import os
from tqdm import tqdm
import glob
import numpy as np
import json
from torchvision.utils import save_image
from torchmetrics.classification import MulticlassConfusionMatrix


from eval.utils.util import batched_inference, run_eval_script
from framework.configs import load_configs_from_logs, adapt_configs_for_inference
from framework.util.load_ckpoint import load_from_disk
from semantic.components.metrics import (
    semantic_accuracy,
    semantic_mIoU,
    confusion_matrix,
    plot_confusion_matrix,
    uncertainty_at_transient,
)


def eval_semantic_nerfs(
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

    output_dp = os.path.join(output_dp, cfgs.run.run_name, "eval_semantic", split)
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

    labels = dataset.semantic_cls_labels.values()
    conf_metric_all = MulticlassConfusionMatrix(num_classes=len(labels), normalize="true")

    for img_idx in progress_bar:
        img = dataset[img_idx]
        rays = img["rays"].to(cuda_device)
        extras = img["extras"].to(cuda_device)
        semantic_gt = img["semantic"].to(cuda_device)
        semantic_gt_no_cars = img["semantic_no_cars"].to(cuda_device)

        results = batched_inference(cfgs, pipeline.renderer, models, rays, extras)

        # update the conf matrix tracking stats for the whole split
        typ = "fine" if "rgb_fine" in results else "coarse"
        conf_metric_all.update(
            results[f"semantic_label_{typ}"].cpu().squeeze(), semantic_gt.cpu().squeeze()
        )
        # additionally create a local matrix to gather stats for the current img
        conf_mat_img_current, conf_mat_val_current = confusion_matrix(
            results, semantic_gt, labels
        )
        # save as png in the eval folder
        conf_mat_current_ofp = os.path.join(output_dp, img["name"] + ".png")
        save_image(conf_mat_img_current, conf_mat_current_ofp)

        all[img["name"]] = {
            "semantic_accuracy": float(semantic_accuracy(results, semantic_gt)),
            "semantic_accuracy_wo_cars": float(
                semantic_accuracy(results, semantic_gt_no_cars)
            ),
            "mIoU": float(semantic_mIoU(conf_mat_val_current.numpy())),
            "uncertainty_at_transient": float(
                uncertainty_at_transient(results, semantic_gt, dataset.car_cls_idx)
            ),
            "confusion_matrix": conf_mat_val_current.numpy().tolist(),
        }

        if "corrupted" in cfgs.pipeline.semantic_dataset_type:
            # if the model were trained on corrupted labels
            # additionally evaluate by comparing to non-corrupted labels
            # to test how well the model is able to recover single details
            semantic_non_corrupted = img["semantic_non_corrupted"].to(cuda_device)
            all[img["name"]].update(
                {
                    "semantic_accuracy_comparison_non_corrupted": float(
                        semantic_accuracy(results, semantic_non_corrupted)
                    ),
                    "semantic_accuracy_comparison_non_corrupted_wo_cars": float(
                        semantic_accuracy(
                            results,
                            semantic_non_corrupted,
                            filter_idx=dataset.car_cls_idx,
                        )
                    ),
                }
            )

        # calculate the average values over the metrics
        metrics = {
            "semantic_accuracy": "Semantic Accuracy (Mean)",
            "semantic_accuracy_wo_cars": "Semantic Accuracy with no cars (Mean)",
            "mIoU": "mIoU (Mean)",
            "semantic_accuracy_comparison_non_corrupted": "Semantic Accuracy comparison to GT (Mean)",
            "semantic_accuracy_comparison_non_corrupted_wo_cars": "Semantic Accuracy comparison to GT w/o cars (Mean)",
            "uncertainty_at_transient": "Uncertainty at transient (Mean)",
        }
        metrics = {k: {"display_name": v, "value": 0.0} for k, v in metrics.items()}
        for value in all.values():
            for metric in metrics:
                if metric in value:
                    metrics[metric]["value"] += float(value[metric])
        for metric in metrics:
            metrics[metric]["value"] /= len(all)

        stats_fp = os.path.join(output_dp, "results.json")
        with open(stats_fp, "w") as f:
            d = all.copy()
            for metric in metrics:
                d[metrics[metric]["display_name"]] = "{:.4f}".format(
                    metrics[metric]["value"]
                )
            json.dump(d, f, indent=4)

    # additionally store the split conf matrix
    conf_mat_img_all = plot_confusion_matrix(conf_metric_all, labels)
    conf_mat_all_ofp = os.path.join(output_dp, "mean.png")
    save_image(conf_mat_img_all, conf_mat_all_ofp)
    # add to .json
    with open(stats_fp, "w") as f:
        d["confusion_matrix"] = conf_metric_all.compute().numpy().tolist()
        json.dump(d, f, indent=4)

    # print("Complete Evaluation Results")
    # print(json.dumps(d, indent=4))


if __name__ == "__main__":
    import fire

    fire.Fire(
        lambda *args, **kwargs: run_eval_script(eval_semantic_nerfs, *args, **kwargs)
    )
