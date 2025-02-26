import os
import glob
import matplotlib.pyplot as plt
import json
from prettytable import PrettyTable
import numpy as np


def gather_exp(experiment_dp, baseline_dp=None):
    experiments = glob.glob(os.path.join(experiment_dp, "*/eval*/"))
    # remove eval*/ part of path
    experiments = [os.path.dirname(os.path.dirname(x)) for x in experiments]
    # filter duplicate entries (if multiple eval results are available for models)
    experiments = list(set(experiments))
    experiments = list(sorted(experiments))

    # add baseline experiment if given at the beginning
    baseline = False
    if baseline_dp is not None and os.path.exists(baseline_dp):
        baseline = True
        experiments.insert(0, baseline_dp)

    assert (
        len(experiments) > 1
    ), "this methods needs to be run on the eval folder of an experiment containing multiple results"
    run(experiments, baseline=baseline)


def gather_custom():
    experiments = []
    run(experiments, output_name="gather_eval_custom")


def _dump_table_txt(gathered, output_fp, experiments, baseline=False):

    table = PrettyTable()

    _table_column_experiment(gathered, table, baseline)
    _table_column_location(gathered, table)
    # _table_column_sparsity(gathered, table)
    # _table_column(gathered, table, "eval", "PSNR (Mean)", "PSNR (Train | Test")
    # _table_column(gathered, table, "eval", "MAE (Mean)", "MAE (Train | Test")
    # _table_column_satnerf_mae(gathered, table)

    _table_column(gathered, table, "eval_semantic", "mIoU (Mean)", "mIoU (Train | Test")
    _table_column(
        gathered,
        table,
        "eval_semantic",
        "Semantic Accuracy (Mean)",
        "Sem_Acc (Train | Test)",
    )
    _table_column(
        gathered,
        table,
        "eval_semantic",
        "Semantic Accuracy with no cars (Mean)",
        "Sem_Acc to no cars GT (Train | Test)",
    )
    # _table_column(
    #     gathered,
    #     table,
    #     "eval_semantic",
    #     "Semantic Accuracy comparison to GT (Mean)",
    #     "Sem_Acc to GT (Train | Test)",
    # )
    # _table_column(
    #     gathered,
    #     table,
    #     "eval_semantic",
    #     "Semantic Accuracy comparison to GT w/o cars (Mean)",
    #     "Sem_Acc w/o cars to GT (Train | Test)",
    # )
    # _table_column_semantic_cars_accuracy(gathered, table)
    _table_column(
        gathered,
        table,
        "eval_semantic",
        "Uncertainty at transient (Mean)",
        "Uncertainty at transient (Train)",
    )

    # _table_column_corrupted_acc(
    #     gathered,
    #     table,
    #     "/mnt/15TB-NVME/val60188/NeRF/inputs/Own_Annotations/corrupted_pixel_masks_rgb",
    # )

    with open(output_fp, "wt") as fp:
        fp.write(table.get_string())
        fp.write("\n\n\n")
        latex_string = table.get_formatted_string("latex")
        latex_string = latex_string.replace("_", "\\_").replace("|", "/")
        fp.write(latex_string)
        fp.write("\n\n\n")
        fp.write("inputs: \n")
        for exp_dp in experiments:
            name = os.path.join(
                os.path.basename(os.path.dirname(exp_dp)), os.path.basename(exp_dp)
            )
            fp.write(name + "\n")


def run(experiments, output_name="gather_eval", baseline=False):
    output_dp = os.path.dirname(experiments[1 if baseline else 0])
    output_dp = os.path.join(output_dp, output_name)
    os.makedirs(output_dp, exist_ok=True)
    print("Storing results in: ", output_dp)

    gathered = {
        "train": _gather_infos(experiments, "train"),
        "test": _gather_infos(experiments, "test"),
    }

    table_txt_fp = os.path.join(output_dp, "results.txt")
    _dump_table_txt(gathered, table_txt_fp, experiments, baseline=baseline)


def _extract_exp_name(gathered):
    exp_names = []
    for train_name, test_name in zip(gathered["train"], gathered["test"]):
        assert train_name == test_name
        splits = train_name.split("_")
        experiment_name = "unknown"
        for split in reversed(splits):
            if "exp" in split:
                experiment_name = split[split.find("exp") + 3 :]
                break
        exp_names.append(experiment_name)
    return exp_names


def _table_column_experiment(gathered, table, baseline=False):
    exp_names = _extract_exp_name(gathered)
    if baseline:
        exp_names[0] = "Baseline"
    table.add_column("Experiment", exp_names)


def _table_column_location(gathered, table):
    locations = []
    for train_name, test_name in zip(gathered["train"], gathered["test"]):
        location = "None"
        for scene_name in ["JAX", "OMA"]:
            if scene_name in train_name:
                start = train_name.find(scene_name)
                location = train_name[start : start + 7]
        locations.append(location)
    table.add_column("Location", locations)


def _table_column(gathered, table, eval, key, name):
    values = []
    for train_name, test_name in zip(gathered["train"], gathered["test"]):
        if key not in gathered["train"][train_name][eval]:
            return

        train = float(gathered["train"][train_name][eval][key])
        out = f"{train:.03f}"
        test = gathered["test"][train_name][eval].get(key)
        if test:
            test = float(test)
            out += f" | {test:.03f}"
        values.append(out)

    table.add_column(name, values)


def _table_column_semantic_cars_accuracy(gathered, table):
    accuracies = []
    for train_name, test_name in zip(gathered["train"], gathered["test"]):
        accuracy_train = np.array(
            gathered["train"][train_name]["eval_semantic"]["confusion_matrix"]
        )[4, 4]
        accuracy_test = np.array(
            gathered["test"][train_name]["eval_semantic"]["confusion_matrix"]
        )[4, 4]
        accuracies.append(f"{accuracy_train:.03f} | {accuracy_test:.03f}")
    table.add_column("Cars Semantic Accuracy (Train | Test)", accuracies)


def _table_column_sparsity(gathered, table):
    sparsity_names = []
    for train_name, test_name in zip(gathered["train"], gathered["test"]):
        sparsity_idx = train_name.find("sparsity")
        if sparsity_idx > 0:
            sparsity_name = train_name[sparsity_idx + len("sparsity")]
        else:
            sparsity_name = "All"
        sparsity_names.append(sparsity_name)
    table.add_column("Sparsity", sparsity_names)


def _table_column_satnerf_mae(gathered, table):
    # mae test values taken from satnerf paper
    data = {"JAX_004": 1.366, "JAX_068": 1.277, "JAX_214": 1.676, "JAX_260": 1.638}
    values = []
    index = table.field_names.index("Location")
    for row in table.rows:
        values.append(data[row[index]])
    table.add_column("satnerf_mae_test", values)


def _table_column_corrupted_acc(gathered, table, path_annotations_dp):
    index = table.field_names.index("Location")
    values = []
    for row in table.rows:
        location = row[index]
        info_txt_fp = os.path.join(
            path_annotations_dp, location, "semantic_accuracies.json"
        )
        with open(info_txt_fp, "r") as fp:
            d = json.load(fp)
        values.append(f"{d['mean']:.03f}")
    table.add_column("Sem_Acc Corrupted_input vs GT", values)


def _gather_infos(experiments, split="train"):

    gathered = {}
    for experiment_dp in experiments:
        experiment_name = os.path.basename(experiment_dp)
        gathered[experiment_name] = {}

        # load semantic eval results if available
        eval_dp = os.path.join(experiment_dp, "eval_semantic", split)
        eval_fp = os.path.join(eval_dp, "results.json")
        if os.path.exists(eval_fp):
            with open(eval_fp) as fp:
                gathered[experiment_name]["eval_semantic"] = json.load(fp)

        # load eval results if available
        eval_dp = os.path.join(experiment_dp, "eval", split)
        eval_fp = os.path.join(eval_dp, "results.json")
        if os.path.exists(eval_fp):
            with open(eval_fp) as fp:
                gathered[experiment_name]["eval"] = json.load(fp)

    return gathered


if __name__ == "__main__":
    import fire

    fire.Fire({"exp": gather_exp, "custom": gather_custom})
