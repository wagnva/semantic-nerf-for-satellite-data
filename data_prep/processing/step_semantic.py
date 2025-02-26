import os
import glob
import rasterio
import json
import numpy as np

from data_prep.processing.step_base import ProcessingStepBase

from data_prep.prepare_annotations import LABELS


class ProcessingStep(ProcessingStepBase):
    def __init__(self, cfg, step_cfg, state):
        super().__init__(cfg, step_cfg, state)
        self.coco_annotations_dp = step_cfg.get("annotations_dp")
        self.rewrite_train_test_split = step_cfg.get("rewrite_train_test_split", False)

        # asserts making sure everything is configured correctly
        assert (
            self.coco_annotations_dp is not None
        ), "coco_annotations_dp needs to be set as part of the step data"
        assert os.path.exists(self.coco_annotations_dp) and os.path.isdir(
            self.coco_annotations_dp
        ), "coco_annotations_dp not a path to a valid dir"
        assert (
            "DFC2019" in cfg.steps[0].file
        ), "Semantic Data only available for DFC2019 Dataset"

        self.output_dp = cfg.output_dp
        self.location_name = cfg.site.location_name
        self.zone_string = cfg.site.zone_string
        self.root_fp = os.path.join(cfg.output_dp, "root.json")

        # output paths
        self.output_cls_dp = os.path.join(cfg.output_dp, "semantic", "coco")
        self.output_cls_corrupted_dp = os.path.join(
            cfg.output_dp, "semantic", "coco_corrupted"
        )
        self.output_cls_nocars_dp = os.path.join(
            cfg.output_dp, "semantic", "coco_no_cars"
        )

        # input paths
        self.semantic_pixels_idp = os.path.join(
            self.coco_annotations_dp, "pixel_masks", self.location_name
        )
        self.semantic_pixels_corrupted_idp = os.path.join(
            self.coco_annotations_dp, "corrupted_pixel_masks", self.location_name
        )
        self.semantic_pixels_nocars_idp = os.path.join(
            self.coco_annotations_dp, "pixel_masks_no_cars", self.location_name
        )

        # State readouts
        self.dsm_fp = state["dsm_fp"]
        self.tifs_dp = state["tifs_dp"]

    def can_be_skipped(self, cfg, state):
        return False

    def run(self, cfg, state):
        self.convert_from_coco()
        self.update_root_file()

    def update_state(self, cfg, state, has_run):
        state["semantic_cls_dp"] = self.output_cls_dp

    def convert_from_coco(self):
        print("Saving cls tifs in following dir:", self.output_cls_dp)
        os.makedirs(self.output_cls_dp, exist_ok=True)
        os.makedirs(self.output_cls_corrupted_dp, exist_ok=True)
        os.makedirs(self.output_cls_nocars_dp, exist_ok=True)

        all_tifs = glob.glob(os.path.join(self.tifs_dp, "*.tif"))
        for tif_fp in all_tifs:
            self.convert_single_file(self.semantic_pixels_idp, self.output_cls_dp, tif_fp)
            self.convert_single_file(
                self.semantic_pixels_corrupted_idp, self.output_cls_corrupted_dp, tif_fp
            )
            self.convert_single_file(
                self.semantic_pixels_nocars_idp, self.output_cls_nocars_dp, tif_fp
            )

            print("Finished creating .tif semantic mask for:", os.path.basename(tif_fp))

    def convert_single_file(self, input_dp, output_dp, tif_fp):
        name = os.path.basename(tif_fp)[:-8] + "_CLS"
        pixel_mask_ifp = os.path.join(input_dp, name + ".npy")
        cls_ofp = os.path.join(output_dp, name + ".tif")

        if not os.path.exists(pixel_mask_ifp):
            if not self.rewrite_train_test_split:
                assert False, f"Missing semantic class file: {name}"
            return

        mask = np.load(pixel_mask_ifp)

        with rasterio.open(tif_fp, "r") as src:
            rpc = src.tags(ns="RPC")
            gcps = src.gcps
            profile = src.profile
            profile["count"] = 1  # change from rgb to single-channel
        with rasterio.open(cls_ofp, "w", **profile) as dst:
            dst.write(mask[None, :, :])  # (1, H, W)
            dst.update_tags(ns="RPC", **rpc)
            if gcps[1] is not None:
                dst.gcps = gcps

    def update_root_file(self):
        update_root_file(
            self.root_fp,
            self.output_cls_dp,
            self.output_cls_corrupted_dp,
            self.output_cls_nocars_dp,
            self.output_dp,
            self.rewrite_train_test_split,
        )


def update_root_file(
    root_fp,
    output_cls_dp,
    output_cls_corrupted_dp,
    output_cls_nocars_dp,
    output_dp,
    rewrite_train_test_split=False,
):
    with open(root_fp) as f:
        d = json.load(f)
    d["semantic_dp_own"] = os.path.relpath(
        output_cls_dp,
        output_dp,
    )
    d["semantic_dp_own_corrupted"] = os.path.relpath(
        output_cls_corrupted_dp,
        output_dp,
    )
    d["semantic_dp_own_no_cars"] = os.path.relpath(
        output_cls_nocars_dp,
        output_dp,
    )
    d["semantic_cls_labels"] = semantic_mapping_information()

    if rewrite_train_test_split:

        all_semantics = glob.glob(os.path.join(output_cls_dp, "*.tif"))
        all_semantics = [x[:-8] + "_RGB.json" for x in all_semantics]
        all_semantics = [os.path.basename(x) for x in all_semantics]
        n_test = max(1, int(0.15 * len(all_semantics)))
        d["train_split"] = all_semantics[n_test:]
        d["test_split"] = all_semantics[:n_test]

        print(
            "Rewriting Train/Test split to include only images with existing annotations"
        )

    with open(root_fp, "w") as f:
        json.dump(d, f, indent=4)
    print("Updated root.json to include semantic information")


def semantic_mapping_information():
    return {str(value): key for key, value in LABELS.items()}
