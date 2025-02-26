import cv2
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
from torchvision.transforms import CenterCrop, Resize

from framework.visualize import ImageVisualization
from framework.datasets import BaseDataset
from framework.util.other import (
    visualize_image,
    scale_image_for_tensorboard,
    SCALE_IMAGE_WIDTH_PIXELS_SMALL,
)

from semantic.components.metrics import semantic_error, confusion_matrix
from data_prep.prepare_annotations import (
    SEMANTIC_CLASS_COLOR_MAPPING as __SEMANTIC_COLOR_MAPPING,
)

SEMANTIC_CLASS_COLOR_MAPPING = None


def get_semantic_class_color_mapping():
    global SEMANTIC_CLASS_COLOR_MAPPING
    if SEMANTIC_CLASS_COLOR_MAPPING is None:
        SEMANTIC_CLASS_COLOR_MAPPING = torch.tensor(__SEMANTIC_COLOR_MAPPING)
    return SEMANTIC_CLASS_COLOR_MAPPING


class TensorboardSemanticSummaryVisualization(ImageVisualization):

    def __init__(
        self,
        cfgs,
        send_to_tensorboard: bool,
        save_as_tif: bool,
        compare_non_corrupted=False,
    ) -> None:
        super().__init__(cfgs, send_to_tensorboard, save_as_tif)
        self.compare_non_corrupted = compare_non_corrupted

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        semantic_labels = results[f"semantic_label{typ}"].view(H, W)  # (H, W)
        color_mappings = get_semantic_class_color_mapping().to(semantic_labels.device)
        semantic_mapped = color_mappings[semantic_labels.long()]  # [H, W, 3]
        semantic_mapped = semantic_mapped.permute(2, 0, 1).cpu()  # [3, H, W]

        semantic_labels = results[f"semantic_label{typ}"].view(H, W)  # (H, W)
        gt = sample["semantic"].to(semantic_labels.device).view(H, W)  # (H, W)
        if self.compare_non_corrupted:
            gt = (
                sample["semantic_non_corrupted"].to(semantic_labels.device).view(H, W)
            )  # (H, W)

        gt_img = color_mappings[gt.long()]  # [H, W, 3]
        gt_img = gt_img.permute(2, 0, 1).cpu()  # [3, H, W]
        error = semantic_error(semantic_labels, gt)
        error_img = visualize_image(error, cmap=cv2.COLORMAP_BONE) * 255.0

        semantic_mapped = scale_image_for_tensorboard(
            semantic_mapped, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL
        )
        gt_img = scale_image_for_tensorboard(gt_img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL)
        error_img = scale_image_for_tensorboard(
            error_img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL
        )

        stack = torch.stack([gt_img, semantic_mapped, error_img])  # (3, 3, H, W)
        stack = stack.to(torch.uint8)

        return stack

    def _name(self) -> str:
        name = "semantic_summary"
        if self.compare_non_corrupted:
            name += "_non_corrupted"
        return name

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # skip any kind of modification for this vis,
        # results from _visualize are already made for tensorboard
        return img


class TensorboardSemanticClassVisualization(ImageVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):

        semantic_logits = results[f"semantic_logits{typ}"].view(
            H, W, -1
        )  # (H, W, N_classes)

        imgs = []
        for class_idx, name in dataset.semantic_cls_labels.items():

            semantic_class_img = semantic_logits[:, :, int(class_idx)]  # (H, W)
            semantic_class_img = (
                visualize_image(semantic_class_img, cmap=cv2.COLORMAP_BONE) * 255.0
            )

            semantic_class_img = scale_image_for_tensorboard(
                semantic_class_img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL
            )
            imgs.append(semantic_class_img)

        stack = torch.stack(imgs)  # (N_classes, 3, H, W)
        stack = stack.to(torch.uint8)

        return stack

    def _name(self) -> str:
        return "semantic_class_overview"

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # skip any kind of modification for this vis,
        # results from _visualize are already made for tensorboard
        return img


class SemanticColorVisualization(ImageVisualization):

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        semantic_labels = results[f"semantic_label{typ}"].view(H, W)  # (H, W)
        color_mappings = get_semantic_class_color_mapping().to(semantic_labels.device)
        semantic_mapped = color_mappings[semantic_labels.long()]  # [H, W, 3]
        semantic_mapped = semantic_mapped.to(torch.uint8)
        semantic_mapped = semantic_mapped.permute(2, 0, 1).cpu()  # [3, H, W]
        return semantic_mapped

    def _name(self) -> str:
        return "semantic_rendering"


class SemanticColorShadingVisualization(ImageVisualization):

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        semantic_labels = results[f"semantic_label{typ}"].view(H, W)  # (H, W)
        color_mappings = get_semantic_class_color_mapping().to(semantic_labels.device)
        semantic_mapped = color_mappings[semantic_labels.long()]  # [H, W, 3]

        # calculate a shading map based on the sun shading scalars
        sun_scalar = results[f"sun{typ}"]
        sun_shading_img = torch.sum(
            results[f"weights{typ}"].unsqueeze(-1) * sun_scalar,
            -2,
        ).view(
            H, W, 1
        )  # (H, W, 1)
        img = semantic_mapped * sun_shading_img  # [H, W, 3]
        img = img.to(torch.uint8)
        return img.permute(2, 0, 1).cpu()  # [3, H, W]

    def _name(self) -> str:
        return "semantic_rendering_shaded"


class SemanticErrorVisualization(ImageVisualization):

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        semantic_labels = results[f"semantic_label{typ}"].view(H, W)  # (H, W)
        gt = sample["semantic"].to(semantic_labels.device).view(H, W)  # (H, W)
        error = semantic_error(semantic_labels, gt)
        return error.cpu()  # [H, W]

    def _name(self) -> str:
        return "semantic_error"

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_BONE


class ConfusionMatrixVisualization(ImageVisualization):

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        labels = dataset.semantic_cls_labels.values()
        conf_matrix_img, _ = confusion_matrix(
            results, sample["semantic"], labels
        )  # (3, H, W)
        return conf_matrix_img

    def _name(self) -> str:
        return "confusion_matrix"


class TensorboardDinoSummaryVisualization(ImageVisualization):

    def __init__(
        self, dataset_train, cfgs, send_to_tensorboard: bool, save_as_tif: bool
    ) -> None:
        super().__init__(cfgs, send_to_tensorboard, save_as_tif)
        self.dataset_train = dataset_train

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        dino_predicted = torch.from_numpy(
            minmax_scale(results[f"dino{typ}"].cpu().numpy())
        )  # (H * W, 3) in range [0, 1]
        dino_predicted = dino_predicted.view(H, W, -1)  # (H, W, 3)
        predicted_img = visualize_dino_features(
            dataset.pca, dino_predicted, H, W
        )  # [H, W, 3]
        predicted_img = predicted_img.permute(2, 0, 1)  # [3, H, W]

        # average over each patch
        dino_mapping = sample["dino_mapping"]
        predicted_img_averaged = torch.zeros_like(results[f"dino{typ}"])
        for idx in torch.unique(dino_mapping):
            local_mask = dino_mapping == idx
            mapping_idxs = (
                local_mask.nonzero().cpu()
            )  # this returns the indices off all pixels handled in patch <idx>
            averaged_dino = torch.mean(results[f"dino{typ}"][mapping_idxs], dim=0)
            predicted_img_averaged[mapping_idxs] = averaged_dino
        predicted_img_averaged = torch.from_numpy(
            minmax_scale(predicted_img_averaged)
        )  # (H * W, 3) in range [0, 1]
        predicted_img_averaged *= 255  # (H, W, 3) in range [0, 255]
        predicted_img_averaged = predicted_img_averaged.to(torch.uint8)
        predicted_img_averaged = predicted_img_averaged.view(H, W, -1).permute(
            2, 0, 1
        )  # [3, H, W]

        # the ground truth dino features are for a possibly padded image
        # therefore use the index pixel mapping to filter to relevant points
        dino_gt = sample["dino"].view(
            sample["dino_h"], sample["dino_w"], -1
        )  # (H^/14, W^/14, N_features)
        gt_img = visualize_dino_features(
            dataset.pca, dino_gt, sample["dino_h"], sample["dino_w"]
        )  # [H^/14, W^/14, 3]
        gt_img = gt_img.permute(2, 0, 1)  # [3, H^/14, W^/14]

        # upsample to remove 14x14 Patching
        gt_img = gt_img.repeat_interleave(
            14 // dataset.dino_upscale, -1
        )  # [3, (H^/14), W^]
        gt_img = gt_img.repeat_interleave(14 // dataset.dino_upscale, -2)  # [3, H^, W^]

        if 14 % dataset.dino_upscale != 0:
            # this can happen for some scales (ex. s=4, 14/4=3.5)
            # in this case resize using interpolation
            gt_img = Resize(
                (
                    int(sample["dino_h"] * (14 / dataset.dino_upscale)),
                    int(sample["dino_w"] * (14 / dataset.dino_upscale)),
                )
            )(gt_img)

        # add padding to make sure its the exact same size as RGB input
        gt_img = CenterCrop([H, W])(gt_img)  # (3, H, W)

        predicted_img = scale_image_for_tensorboard(
            predicted_img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL
        )
        predicted_img_averaged = scale_image_for_tensorboard(
            predicted_img_averaged, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL
        )
        gt_img = scale_image_for_tensorboard(gt_img, size=SCALE_IMAGE_WIDTH_PIXELS_SMALL)

        stack = torch.stack(
            [gt_img, predicted_img_averaged, predicted_img]
        )  # (2, 3, H, W)
        stack = stack.to(torch.uint8)

        return stack

    def _name(self) -> str:
        return "dino_summary"

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # skip any kind of modification for this vis,
        # results from _visualize are already made for tensorboard
        return img


def visualize_dino_features(pca, dino_tensor, h=None, w=None):

    if dino_tensor.shape[-1] > 3:

        if torch.is_tensor(dino_tensor):
            dino_tensor = dino_tensor.cpu().numpy()

        # apply trained PCA to input
        dino_tensor = pca.transform(dino_tensor.reshape(-1, dino_tensor.shape[-1]))
        dino_tensor = minmax_scale(dino_tensor)

    if not torch.is_tensor(dino_tensor):
        dino_tensor = torch.from_numpy(dino_tensor)

    # scale each feature to (0,1) and then convert to [0, 255]
    if torch.max(dino_tensor) <= 1.2:
        dino_tensor *= 255.0
        dino_tensor = dino_tensor.to(torch.uint8)

    # reshape the features to the original image size
    if h is not None and w is not None:
        dino_tensor = dino_tensor.reshape([h, w, 3])

    return dino_tensor


class NeighbourmaskVisualization(ImageVisualization):
    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        mask = results[f"neighbour_mask{typ}"].cpu()  # (H*W)
        image = torch.ones((H * W))  # (H * W, 3)
        image[~mask] *= 0.0
        return image.view((H, W))

    def _name(self) -> str:
        return "neighbour_smoothing_mask"

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_BONE


class DepthsRegVisualization(ImageVisualization):

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        depths = results[f"neighbours{typ}"].cpu()  # (N_r, N_Neighbours)

        main_depth = depths[:, 0]  # [N]
        other_depths = depths[:, 1:]  # [N, N_Neighbours]
        mean_depth = torch.mean(other_depths, dim=-1)  # [N]
        diff = torch.abs(main_depth - mean_depth) ** 2  # (N]

        image = torch.zeros((H * W))  # (H*W)
        mask = results[f"neighbour_mask{typ}"].cpu()  # (H*W)
        image[mask] = diff

        return image.view(H, W)

    def _name(self) -> str:
        return "depths_reg"

    def _get_visualize_color_scheme(self):
        return cv2.COLORMAP_BONE


class DensityRegVisualization(ImageVisualization):

    def __init__(self, cfgs, send_to_tensorboard, save_as_tif, apply_to_labels=[0, 1]):
        super().__init__(cfgs, send_to_tensorboard, save_as_tif)
        self.apply_to_labels = apply_to_labels

    def _visualize(self, pipeline, dataset: BaseDataset, sample, results, W, H, typ):
        mean_sigma = results[f"neighbour_mean_sigma{typ}"].cpu()  # (H*W, 2)
        neighbour_mask = mean_sigma[:, 2].bool()  # [H*W]
        main_sigma = mean_sigma[:, 1]  # [H*W]
        mean_sigma = mean_sigma[:, 0]  # [H*W]

        difference = torch.square(torch.abs(mean_sigma - main_sigma))  # [H*W]

        # apply masks
        labels = results[f"semantic_label{typ}"].cpu()

        if isinstance(self.apply_to_labels, list):
            self.apply_to_labels = torch.tensor(
                self.apply_to_labels, device=labels.device
            )

        mask = torch.isin(labels, self.apply_to_labels)
        mask = torch.logical_and(mask, neighbour_mask)
        difference[~mask] *= 0.0

        difference = difference.view(H, W)

        difference_img = visualize_image(difference, cmap=cv2.COLORMAP_BONE) * 255.0
        mask_img = visualize_image(mask.view(H, W).int(), cmap=cv2.COLORMAP_BONE) * 255.0

        return torch.stack([difference_img, mask_img])  # [2, 3, H, W]

    def _name(self) -> str:
        return "density_reg"

    def _visualize_image_for_tensorboard(self, img: torch.tensor, W, H) -> np.ndarray:
        # skip any kind of modification for this vis,
        # results from _visualize are already made for tensorboard
        return img
