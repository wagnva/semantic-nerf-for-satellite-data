import json

import numpy as np
import os
import glob
import shutil
from pycocotools.coco import COCO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
import tqdm
import numba


LABELS = {"ground": 0, "water": 1, "vegetation": 2, "buildings": 3, "cars": 4}
LOCATIONS = ["JAX_004", "JAX_068", "JAX_214", "JAX_260"]
DEFAULT_CLASS = {
    "JAX_004": "vegetation",
    "JAX_068": "ground",
    "JAX_214": "ground",
    "JAX_260": "ground",
}  # which class to use for unlabeled pixels
SEMANTIC_CLASS_COLOR_MAPPING = np.array(
    [
        [229, 232, 157],  # light yellow
        [35, 161, 228],  # light blue
        [9, 171, 120],  # green
        [138, 138, 138],  # light gray
        [193, 79, 69],  # red
        [98, 98, 98],  # dark gray
    ],
    dtype=np.uint8,
)

# settings for the corrupted labels
CORRUPT_BORDER_GROWTH = {
    "ground": 10,
    "water": 0,
    "vegetation": 10,
    "buildings": 10,
    "cars": 0,
}  # how much of the border area is considered for each class
CORRUPT_HOW_MUCH_ACC = {
    "ground": 0.2,
    "water": 0.2,
    "vegetation": 0.2,
    "buildings": 0.2,
    "cars": 0.2,
}  # percentage of pixels modulated across whole mask
CORRUPT_HOW_MUCH_ACC_BORDERS = {
    "ground": 0.0,
    "water": 0.0,
    "vegetation": 0.0,
    "buildings": 0.0,
    "cars": 0.0,
}  # percentage of pixels modulated across border
CORRUPT_REPLACE_WITH = ["ground", "buildings", "vegetation"]


def main(roboflow_dp, corrupt=False, no_cars=True):
    # preprocess the unsorted roboflow files so that each location is in its own folder
    # additionally strips prefixes added to some image files by roboflow
    output_preprocessed = os.path.join(os.path.dirname(roboflow_dp), "roboflow_sorted")
    os.makedirs(output_preprocessed, exist_ok=True)
    roboflow_image_dp = os.path.join(roboflow_dp, "train")
    preprocess_from_roboflow(roboflow_image_dp, output_preprocessed)

    # convert into bitmasks
    output_bitmasks = os.path.join(os.path.dirname(roboflow_dp), "pixel_masks")
    output_viz = os.path.join(os.path.dirname(roboflow_dp), "pixel_masks_rgb")
    output_bitmasks_noise = os.path.join(
        os.path.dirname(roboflow_dp), "corrupted_pixel_masks"
    )
    output_viz_noise = os.path.join(
        os.path.dirname(roboflow_dp), "corrupted_pixel_masks_rgb"
    )
    output_bitmasks_nc = os.path.join(os.path.dirname(roboflow_dp), "pixel_masks_no_cars")
    output_viz_nc = os.path.join(os.path.dirname(roboflow_dp), "pixel_masks_rgb_no_cars")
    convert_to_pixelmasks(
        output_preprocessed,
        output_bitmasks,
        output_viz,
        output_bitmasks_noise,
        output_viz_noise,
        output_bitmasks_nc,
        output_viz_nc,
        create_corrupt=corrupt,
        create_no_cars=no_cars,
    )


def preprocess_from_roboflow(roboflow_dp, output_dp):

    # copy the coco annotations file
    shutil.copy(
        os.path.join(roboflow_dp, "_annotations.coco.json"),
        os.path.join(output_dp, "_annotations.coco.json"),
    )

    for location in LOCATIONS:
        image_files = glob.glob(os.path.join(roboflow_dp, f"*{location}*.jpg"))
        location_output_dp = os.path.join(output_dp, location)
        os.makedirs(location_output_dp, exist_ok=True)
        for image_fp in image_files:
            image_name = cleanup_roboflow_img_name(os.path.basename(image_fp))
            shutil.copy(image_fp, os.path.join(location_output_dp, image_name + ".jpg"))
    print("Preprocessed unsorted Roboflow images into separate folders for each location")


def convert_to_pixelmasks(
    semantic_input_dp,
    output_bitmasks_dp,
    output_viz_dp,
    output_bitmasks_noise_dp,
    output_viz_noise_dp,
    output_bitmasks_nc_dp,
    output_viz_nc_dp,
    create_corrupt=False,
    create_no_cars=False,
):
    coco = COCO(os.path.join(semantic_input_dp, "_annotations.coco.json"))
    all_coco_images = {
        cleanup_roboflow_img_name(img["file_name"]): img for img in coco.imgs.values()
    }

    for index, location in enumerate(LOCATIONS):

        print(f"Creating pixelmasks for location '{location}'")

        location_image_fps = glob.glob(os.path.join(semantic_input_dp, location, "*.jpg"))
        location_image_names = [os.path.basename(x)[:-4] for x in location_image_fps]

        output_bitmasks_location_dp = os.path.join(output_bitmasks_dp, location)
        os.makedirs(output_bitmasks_location_dp, exist_ok=True)
        output_viz_location_dp = os.path.join(output_viz_dp, location)
        os.makedirs(output_viz_location_dp, exist_ok=True)

        # corrupt locations
        output_bitmasks_location_noise_dp = os.path.join(
            output_bitmasks_noise_dp, location
        )
        output_viz_location_noise_dp = os.path.join(output_viz_noise_dp, location)
        if create_corrupt:
            os.makedirs(output_bitmasks_location_noise_dp, exist_ok=True)
            os.makedirs(output_viz_location_noise_dp, exist_ok=True)
        # no cars location
        output_bitmasks_location_nc_dp = os.path.join(output_bitmasks_nc_dp, location)
        output_viz_location_nc_dp = os.path.join(output_viz_nc_dp, location)
        if create_no_cars:
            os.makedirs(output_bitmasks_location_nc_dp, exist_ok=True)
            os.makedirs(output_viz_location_nc_dp, exist_ok=True)

        corrupt_semantic_accuracy = {}
        with tqdm.tqdm(total=len(location_image_names), desc=str(location)) as pb:
            for img_name, img in all_coco_images.items():

                if not img_name in location_image_names:
                    continue

                mask = get_mask_for_img(coco, img["id"], DEFAULT_CLASS[location])
                store_pixel_mask(
                    mask, img_name, output_bitmasks_location_dp, output_viz_location_dp
                )

                if create_no_cars:
                    mask_nc = get_mask_for_img(
                        coco, img["id"], DEFAULT_CLASS[location], no_cars=True
                    )
                    store_pixel_mask(
                        mask_nc,
                        img_name,
                        output_bitmasks_location_nc_dp,
                        output_viz_location_nc_dp,
                    )

                if create_corrupt:
                    mask_noise = get_mask_for_img(
                        coco, img["id"], DEFAULT_CLASS[location], corrupt=True
                    )
                    store_pixel_mask(
                        mask_noise,
                        img_name,
                        output_bitmasks_location_noise_dp,
                        output_viz_location_noise_dp,
                    )
                    corrupt_semantic_accuracy[img_name] = semantic_accuracy(
                        mask_noise, mask, filter_idx=4  # "cars"
                    )

                pb.update(1)

        if create_corrupt:
            corrupt_semantic_accuracy["mean"] = sum(
                corrupt_semantic_accuracy.values()
            ) / len(corrupt_semantic_accuracy)
            with open(
                os.path.join(output_viz_location_noise_dp, "semantic_accuracies.json"),
                "wt",
            ) as fp:
                json.dump(corrupt_semantic_accuracy, fp, indent=4)


def store_pixel_mask(mask, img_name, output_npy_dp, output_viz_dp):
    cls_name = img_name[:-4] + "_CLS"

    # store the mask as .npy
    output_fp = os.path.join(output_npy_dp, f"{cls_name}.npy")
    np.save(output_fp, mask)

    # additionally store an rgb visualization of the mask as jpg
    mask_viz = SEMANTIC_CLASS_COLOR_MAPPING[mask]
    output_fp = os.path.join(output_viz_dp, f"{cls_name}.jpg")
    img_mask = Image.fromarray(mask_viz)
    img_mask.save(output_fp)


def get_mask_for_img(coco, img_id, default_class="ground", corrupt=False, no_cars=False):
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    mask = np.ones_like(coco.annToMask(anns[0])) * -1

    for i in range(len(anns)):
        category_ranking = remap_labels(coco, anns[i]["category_id"])

        if no_cars and category_ranking == LABELS["cars"]:
            continue

        # always choose the highest category based on ranking
        # to handle mask overlaps (cars should always rank above ground etc.)
        category_mask = coco.annToMask(anns[i]).astype(np.int8)

        # replace default values (0) with -1 so that ranking works
        category_mask[category_mask == 0] = -1

        # relabel 1 to the specified category ranking and replace in complete mask based on ranking
        category_mask[category_mask > 0] = (
            category_mask[category_mask > 0] * category_ranking
        )

        # always place highest value in mask
        mask = np.maximum(mask, category_mask)

    # replace all unlabeled areas (-1) with default class
    default_class_v = LABELS.get(default_class, 0)
    mask[mask < 0] = default_class_v
    # fix dtype after removing negative values
    mask = mask.astype(np.uint8)

    if corrupt:
        mask = corrupt_mask(mask, default_class_v)

    return mask


def corrupt_mask(mask, default_class_v):
    partial_masks = {}
    rng = np.random.default_rng()

    for type, type_v in LABELS.items():

        partial_mask = mask == type_v
        debug_save(partial_mask, f"0_{type_v}_partial_mask")

        corrupt_how_much = CORRUPT_HOW_MUCH_ACC.get(type, 1)
        corrupt_how_much_border = CORRUPT_HOW_MUCH_ACC_BORDERS.get(type, 1)
        if (
            corrupt_how_much == 0 and corrupt_how_much_border == 0
        ) or partial_mask.sum() == 0:
            partial_masks[type_v] = (partial_mask, np.zeros_like(partial_mask))
            continue

        # generate noise
        # create a normal noise map with the same size
        normal_noise = create_noise(rng, mask.shape, True)
        _, _, noise_mask = threshold_mask(normal_noise, partial_mask, corrupt_how_much)

        # debug_save(noise_mask, f"2_{type_v}_noise_mask_remove")
        partial_mask_corrupted = partial_mask.astype(np.float32)
        partial_mask_corrupted[noise_mask] = 0
        debug_save(partial_mask_corrupted, f"1_{type_v}_whole_corrupted")

        print(f"{type}: {(1 - partial_mask_corrupted.sum() / partial_mask.sum()):.02%}")

        # corrupt borders to simulate uncertain along annotation edges
        border_size = CORRUPT_BORDER_GROWTH.get(type, 0)
        if border_size > 0 and corrupt_how_much_border > 0:
            normal_noise = create_noise(rng, mask.shape, False)
            partial_mask_corrupted_bfb = partial_mask_corrupted.copy()
            partial_mask_corrupted = corrupt_mask_border(
                partial_mask_corrupted,
                type_v,
                normal_noise,
                border_size,
                corrupt_how_much_border,
            )
            partial_mask_corrupted = partial_mask_corrupted > 0
            print(
                f"{type} Border: {(1 - partial_mask_corrupted.sum() / partial_mask_corrupted_bfb.sum()):.02%}"
            )

        # save outputs and a separate mask with the removed parts
        partial_mask_corrupted = partial_mask_corrupted > 0
        partial_mask_removed = ~partial_mask_corrupted & partial_mask
        debug_save(partial_mask_removed, f"2_{type_v}_removed")
        partial_masks[type_v] = (partial_mask_corrupted, partial_mask_removed)

    # out = np.ones_like(mask) * 25  # undefined label
    out = np.ones_like(mask) * default_class_v  # default class
    labels = list(
        map(lambda x: LABELS[x], filter(lambda x: x in LABELS, CORRUPT_REPLACE_WITH))
    )
    for type_v, masks in partial_masks.items():
        mask, mask_removed = masks
        out[mask] = type_v

        # fill in the removed parts using random classes for each cluster
        clusters = cluster_mask(mask_removed)
        for removal_cluster in clusters:
            options = [x for x in labels if x != type_v]
            out[removal_cluster] = rng.choice(options)

    debug_save(out, f"9_out")

    return out


def corrupt_mask_border(category_mask, type_v, normal_noise, border_growth, acc_loss):
    if acc_loss == 0:
        return category_mask

    category_mask = category_mask.astype(np.float32)  # change dtype for cv2

    # detect borders
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    borders = cv2.filter2D(src=category_mask, ddepth=-1, kernel=kernel)
    category_mask = category_mask > 0
    # debug_save(borders > 0, f"2_{type_v}_borders")

    # increase its width to a workable amount
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    borders_dilated = cv2.dilate(borders, kernel, iterations=border_growth)
    borders_dilated = borders_dilated > 0
    # debug_save(borders_dilated, f"3_{type_v}_borders_dilated")

    # remove from mask for all randomly selected pixels BELOW threshold
    borders_inside = borders_dilated & category_mask
    noise_mask_remove, _, _ = threshold_mask(normal_noise, borders_inside, acc_loss)
    noise_mask_remove &= borders_dilated
    category_mask[noise_mask_remove] = 0

    # remove from mask for all randomly selected pixels ABOVE threshold
    borders_outside = borders_dilated & (~category_mask)
    _, noise_mask_add, _ = threshold_mask(normal_noise, borders_outside, acc_loss)
    noise_mask_add &= borders_dilated
    category_mask[noise_mask_add] = 1

    # smooth by erosion and dilation
    # category_mask = category_mask.astype(np.float32)
    # for i in range(6):
    #     category_mask = cv2.dilate(category_mask, kernel, iterations=1)
    #     category_mask = cv2.erode(category_mask, kernel, iterations=1)

    out = np.zeros(category_mask.shape)
    out[category_mask > 0] = 1

    debug_save(out, f"4_{type_v}_borders_corrupted")

    return out


def create_noise(rng, shape, large=False):

    sigmaX, sigmaY = 16, 16
    if large:
        sigmaX *= 4
        sigmaY *= 4

    # create a normal noise map with the same size
    normal_noise = rng.random(shape)
    # blur
    normal_noise = cv2.GaussianBlur(
        normal_noise,
        (0, 0),
        sigmaX=sigmaX,
        sigmaY=sigmaY,
        borderType=cv2.BORDER_DEFAULT,
    )
    # rescale to [0, 1] (as blurring shifts towards 0.5)
    normal_noise = (normal_noise - np.min(normal_noise)) / (
        np.max(normal_noise) - np.min(normal_noise)
    )
    return normal_noise


def threshold_mask(noise, mask, threshold_amount):
    # only select parts of noise map of mask area
    flat_noise = noise.flatten()
    flat_noise = flat_noise[mask.flatten()]

    # determine bounds so that specified accuracy loss is achieved for class
    flat_noise = np.sort(flat_noise)
    thresh_c = int(flat_noise.size * threshold_amount * 0.5)
    thresh_low = flat_noise[thresh_c - 1]
    thresh_high = flat_noise[-thresh_c]

    # apply thresholds
    # this contains roughly CORRUPT_HOW_MUCH_ACC percentage of pixels
    below = noise < thresh_low
    above = noise > thresh_high
    noise_mask = below | above
    return below, above, noise_mask


def cluster_mask(mask):
    mask = mask.astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # separate into its own boolean masks
    cluster_masks = []
    for contour in contours:
        cluster_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(cluster_mask, [contour], -1, 255, thickness=cv2.FILLED)
        cluster_masks.append(cluster_mask.astype(bool))

    return cluster_masks


def determine_mask_shortest_side(mask):
    if not mask.any():
        return 0

    rows_with_mask = mask.any(axis=1)
    y_min, y_max = np.where(rows_with_mask)[0][[0, -1]]

    cols_with_mask = mask.any(axis=0)
    x_min, x_max = np.where(cols_with_mask)[0][[0, -1]]

    x_extent = x_max - x_min + 1
    y_extent = y_max - y_min + 1
    return min(x_extent, y_extent)


def remap_labels(coco, category_id):
    return LABELS.get(coco.cats[category_id]["name"], 0)


def cleanup_roboflow_img_name(image_name):
    image_name = image_name.split(".")[0][:-4]  # remove added postfix and file ending
    image_name = image_name[image_name.find("JAX") :]  # remove possible numbered prefix
    return image_name


def debug_save(img, name, save=False):
    if save:
        DEBUG_DIR = "/mnt/15TB-NVME/val60188/NeRF/inputs/Own_Annotations/corrupted_debug"
        ofp = os.path.join(DEBUG_DIR, name + ".jpg")
        plt.imsave(ofp, img, cmap="bone")


def semantic_accuracy(mask, mask_corrupted, filter_idx=None):
    mask_ = mask.flatten()
    mask_corrupted_ = mask_corrupted.flatten()
    # 0 if correct class, 1 if wrong class
    error = np.clip(np.abs(mask_ - mask_corrupted_), a_min=0.0, a_max=1.0)

    # filter specific index
    if filter_idx is not None:
        tmp_mask = mask_ == filter_idx
        error[tmp_mask] *= 0

    return 1 - (np.sum(error, axis=0) / mask.size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
