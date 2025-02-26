import torch
import io
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def semantic_error(semantic_pred, semantic_gt, filter_idx=None):
    semantic_gt_ = semantic_gt.flatten()
    semantic_pred_ = semantic_pred.flatten()
    # 0 if correct class, 1 if wrong class
    error = torch.clamp(torch.abs(semantic_gt_ - semantic_pred_), min=0.0, max=1.0)

    # filter specific index
    if filter_idx is not None:
        mask = semantic_gt_ == filter_idx
        error[mask] *= 0

    return error.reshape(semantic_gt.shape)


def semantic_accuracy(results, targets, filter_idx=None):
    typ = "fine" if "rgb_fine" in results else "coarse"
    semantic_pred = results[f"semantic_label_{typ}"]
    error = semantic_error(semantic_pred, targets, filter_idx=filter_idx).flatten()
    return 1 - (torch.sum(error, dim=0) / len(targets))


def semantic_mIoU(confusion_matrix_values):
    # Source: https://github.com/Harry-Zhi/semantic_nerf/blob/b79f9c3640b62350e9c167a66c273c2121428ce1/SSR/training/training_utils.py#L77C5-L82C25
    number_classes = confusion_matrix_values.shape[0]
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = confusion_matrix_values[class_id, class_id] / (
            np.sum(confusion_matrix_values[class_id, :])
            + np.sum(confusion_matrix_values[:, class_id])
            - confusion_matrix_values[class_id, class_id]
        )
    miou = np.nanmean(ious)
    return miou
    # typ = "fine" if "rgb_fine" in results else "coarse"
    # semantic_pred = results[f"semantic_label_{typ}"].flatten()  # (H*W)
    # targets = targets.flatten()  # (H*W)
    # # targets dtype is uint8, torchmetrics expects int64
    # targets = targets.to(torch.int64)
    #
    # n_classes = results[f"semantic_logits_{typ}"].shape[1]
    # mIoU = MeanIoU(n_classes).to(targets.device)
    # return mIoU(semantic_pred[None, ...], targets[None, ...])


def confusion_matrix(results, targets, labels):
    typ = "fine" if "rgb_fine" in results else "coarse"
    semantic_pred = results[f"semantic_label_{typ}"]
    metric = MulticlassConfusionMatrix(num_classes=len(labels), normalize="true")
    values = metric(semantic_pred.cpu().squeeze(), targets.cpu().squeeze())
    return plot_confusion_matrix(metric, labels), values


def plot_confusion_matrix(metric, labels):
    fig, ax = plt.subplots(figsize=(10, 10))

    metric.plot(ax=ax, labels=labels)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()

    img_w_alpha = transforms.ToTensor()(Image.open(buf))  # (4, H, W)
    img = img_w_alpha[:3, :, :]  # (3, H, W)

    return img


def uncertainty_at_transient(results, semantic_gt, car_idx):
    typ = "fine" if "rgb_fine" in results else "coarse"
    beta = results[f"beta_{typ}"]  # (N_rays, N_samples)
    weights = results[f"weights_{typ}"]  # (N_rays, N_samples)
    beta = torch.sum(weights.unsqueeze(-1) * beta, -2)  # (N_rays, 1)
    semantic_mask = semantic_gt == car_idx
    semantic_mask = semantic_mask.flatten().cpu()
    beta_sum = beta[semantic_mask].sum() / semantic_mask.sum()
    return beta_sum
