from textwrap import indent

import toml
import os
from datetime import datetime
import subprocess

from framework.configs import load_configs
from framework.util.train_util import get_list_of_free_cuda_devices
from framework.logger import logger


def run_automated_training_tmux(experiment_cfg_fp, env_name="rs"):
    # load experiments config and create experiment folder
    experiment_cfg = toml.load(experiment_cfg_fp)

    # assume the <experiment>.toml has the path */configs/experiments/*/<experiment>.toml
    base_dir_name = os.path.join("configs", "experiments")
    base_dir_find = experiment_cfg_fp.find(base_dir_name)
    experiment_cfg["cfgs_base_dir"] = os.path.dirname(
        experiment_cfg_fp[: base_dir_find + len(base_dir_name)]
    )
    run_cfg = toml.load(
        os.path.join(experiment_cfg["cfgs_base_dir"], experiment_cfg["run_cfg"])
    )

    # get the experiment category either from the run config or from the [run] subgroup in the experiment cfg
    experiment_category = experiment_cfg.get("run", {}).get(
        "experiment_category", run_cfg["experiment_category"]
    )
    experiment_name = experiment_cfg["experiment_name"]
    # if the experiment name starts with an underscore, assume the name is fixed and dont prepend with timestamp
    if not experiment_name.startswith("_"):
        timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "_" + timestr + "_" + experiment_name
    output_dp = os.path.join(
        run_cfg["workspace_dp"], "_" + experiment_category, experiment_name
    )
    output_dp_cfgs = os.path.join(output_dp, ".cfgs")
    os.makedirs(output_dp_cfgs, exist_ok=True)

    # read in experiments and write modified cfgs
    ids = convert_experiments_to_cfgs(
        output_dp_cfgs, experiment_cfg, experiment_category, experiment_name
    )
    # get the allowed gpus either from the run config or from the [run] subgroup in the experiment cfg
    allowed_gpus = experiment_cfg.get("run", {}).get("gpu_id", run_cfg["gpu_id"])
    available_devices = get_list_of_free_cuda_devices(allowed=allowed_gpus)
    ids_assigned_to_gpus = assign_ids_to_gpus(ids, available_devices)

    # create tmux session creation script
    tmux_sh_fp, session_name = create_tmux_session_script(
        output_dp_cfgs, experiment_name, ids_assigned_to_gpus, env_name
    )

    # run tmux creation script in a new terminal window and then attach the session
    subprocess.call(
        [
            "gnome-terminal",
            "--",
            "bash",
            "-c",
            f"bash {tmux_sh_fp}; tmux a -t {session_name}",
        ]
    )


def assign_ids_to_gpus(ids, gpus):
    if len(gpus) < len(ids):
        logger.info(
            "Automated_Training",
            "Trying to start more trainings than available GPUs. Training will be performed sequential.",
        )
        logger.info(
            "Automated_Training",
            f"Subset started directly: {', '.join(ids[:len(gpus)])}, sequential later on: {', '.join(ids[len(gpus):])}",
        )

    out = [[] for x in range(min(len(ids), len(gpus)))]
    for idx, id in enumerate(ids):
        out[idx % len(gpus)].append(id)
    return {gpus[idx]: ids for idx, ids in enumerate(out)}


def convert_experiments_to_cfgs(
    output_dp, experiment_cfg, experiment_category, experiment_name
):
    experiments = {}
    run_ids = experiment_cfg.get("run_ids", None)
    for cfg in experiment_cfg["experiments"]:
        id = cfg["id"]
        assert id not in list(experiments.keys()), "experiment id has to be unique"
        if run_ids and id not in run_ids:
            continue
        experiments[id] = cfg

    for id, experiment in experiments.items():
        loaded_cfg = load_configs(
            os.path.join(experiment_cfg["cfgs_base_dir"], experiment_cfg["run_cfg"]),
            os.path.join(experiment_cfg["cfgs_base_dir"], experiment["pipeline_name"]),
        )

        # set default dataset_name if one is given
        # can be overwritten for a single experiment in the run cfg part
        # if neither is set, the training will handle throwing an error on startup
        loaded_cfg.run.dataset_name = experiment_cfg.get("dataset_name", "")

        # update values based on settings for complete experiment
        for cfg_name in ["pipeline", "run"]:
            if experiment_cfg.get(cfg_name):
                for key, value in experiment_cfg.get(cfg_name).items():
                    assert hasattr(
                        getattr(loaded_cfg, cfg_name), key
                    ), f"Trying to overwrite missing cfg entry '{cfg_name}.{key} = {value}'"
                    setattr(getattr(loaded_cfg, cfg_name), key, value)

        # update values based on specific experiment changes
        for cfg_name in ["pipeline", "run"]:
            if experiment.get(cfg_name):
                for key, value in experiment[cfg_name].items():
                    assert hasattr(
                        getattr(loaded_cfg, cfg_name), key
                    ), f"Trying to overwrite missing cfg entry '{cfg_name}.{key} = {value}'"
                    setattr(getattr(loaded_cfg, cfg_name), key, value)

        loaded_cfg.run.experiment_category = os.path.join(
            experiment_category, experiment_name
        )
        loaded_cfg.run.run_name_postfix += f"_exp{id}"

        # store changed configs
        loaded_cfg.dump_to_toml(output_dp, f"{id}_pipeline", f"{id}_run")

    # return the names of all experiments that were prepared
    return list(experiments.keys())


def create_tmux_session_script(output_dp, name, ids_to_gpus, env_name="rs"):
    timestr = datetime.now().strftime("%m-%d_%H-%M")
    script_lines = [
        "#!/bin/bash",
        f"session='{timestr}'",
        "tmux new-session -d -s $session",
        "tmux rename-window -t $session:0 'overview'",
        # enable scroll bars (weird behavior when switching windows - disabled for now)
        # "tmux set -ga terminal-overrides ',xterm*:smcup@:rmcup@'"
        "tmux setw -g mouse on",
    ]

    n_experiments = sum([len(x) for x in ids_to_gpus.values()])
    intro_text = [
        f'Experiment "{name}"',
        f"with {n_experiments} experiments on {len(ids_to_gpus.keys())} gpu(s): "
        + ", ".join([str(x) for x in ids_to_gpus.keys()]),
    ]
    header = "\n  " + "#" * max([len(x) for x in intro_text]) + "  \n"
    intro_text.insert(0, header)
    intro_text.append(header)

    intro_text.append("Device    Experiment ID(s)")
    for gpu, ids in ids_to_gpus.items():
        id_string = ", ".join([f'"{x}"' for x in ids])
        intro_text.append(f"cuda_{gpu} => {id_string}")

    intro_text.append("\n")
    intro_text = ["  " + x + "  " for x in intro_text]
    intro_text = "\n".join(intro_text)
    intro_script = ["clear", f"echo '{intro_text}'"]

    tmux_sh_intro_fp = os.path.join(output_dp, "tmux_intro_text.sh")
    with open(tmux_sh_intro_fp, "wt") as fp:
        fp.write("\n".join(intro_script))
    script_lines.append(
        f"tmux send-keys -t $session:0 'bash {tmux_sh_intro_fp}' C-m",
    )
    script_lines.append(
        f"tmux send-keys -t $session:0 'nvidia-htop.py'",
    )

    # add a window with tensorboard command prewritten
    script_lines.append("tmux new-window -t $session:1 -n 't'")
    script_lines.append(f"tmux send-keys -t $session:1 'conda activate {env_name}' C-m")
    script_lines.append(
        f"tmux send-keys -t $session:1 'tensorboard --logdir={os.path.dirname(output_dp)} --port=6010'"
    )

    window_id = 2
    for gpu, ids in ids_to_gpus.items():
        script_lines.append(f"tmux new-window -t $session:{window_id} -n 'cuda_{gpu}'")
        script_lines.append(
            f"tmux send-keys -t $session:{window_id} 'conda activate {env_name}; sleep {window_id}; "
            f"python -m run.training start_assigned_ids_from_automated "
            f"{output_dp} {gpu} {' '.join(ids)}' C-m",
        )
        window_id += 1

    script_lines.append("tmux select-window -t $session:0")

    tmux_sh_fp = os.path.join(output_dp, "tmux.sh")
    with open(tmux_sh_fp, "wt") as fp:
        fp.write("\n".join(script_lines))

    return tmux_sh_fp, timestr


if __name__ == "__main__":
    import fire

    fire.Fire(run_automated_training_tmux)
