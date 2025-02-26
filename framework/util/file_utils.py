import os
import json
import numpy as np
import glob
from shutil import copyfile
import sys
import torch


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def save_dict_of_tensors(data, name: str, save_to_dp: str):
    assert os.path.isdir(save_to_dp), "Directory does not exist"
    for key in data:
        if key == "name":
            continue
        assert torch.is_tensor(data[key]), "Trying to save a non tensor item"
        output_fp = os.path.join(save_to_dp, f"{os.path.basename(name)}_{key}.data")
        torch.save(data[key], output_fp)


def load_dict_of_tensors(name: str, load_from_dp: str):
    assert os.path.isdir(load_from_dp), "Directory does not exist"
    files = glob.glob(os.path.join(load_from_dp, f"{name}_*.data"))
    output = {}
    for file in files:
        # the file name is assumed as <name>_<key>.data
        key = os.path.basename(file)[:-5].split("_")[-1]
        output[key] = torch.load(file, weights_only=True)
        output[key] = output[key].type(torch.FloatTensor)
    return output


def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d


def write_dict_to_json(d, output_path):
    with open(output_path, "w") as f:
        json.dump(d, f, indent=4)
    return d


def search_in_python_path(filename):
    """Search for filename in the list of directories specified in the
    PYTHONPATH environment variable.
    """
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        for d in pythonpath.split(os.pathsep):
            filepath = os.path.join(d, filename)
            if os.path.isfile(filepath):
                return filepath
    return None


def assert_fp_exists(fp):
    assert os.path.exists(fp) and os.path.isfile(
        fp
    ), f"Necessary file doesn't exist: {fp}"
