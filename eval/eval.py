import subprocess
import os
import glob

from framework.util.train_util import get_list_of_free_cuda_devices


def start(path, tmux_fp=None, out_dp=None, env_name="rs", gpus=None, tmux="eval_all"):

    if tmux_fp is None:
        tmux_fp = find_tmux_fp()
        print("No tmux script provided as argument.")
        print("Using following tmux script: ")
        print(tmux_fp)

    free_gpus = get_list_of_free_cuda_devices(allowed=gpus)
    gpu_str = " ".join([str(x) for x in free_gpus])

    env = os.environ.copy()
    if out_dp is not None:
        env["SEMANTIC_SATNERF_EVAL_DP"] = out_dp

    # start tmux session
    out = subprocess.run(
        [f"bash {tmux_fp} {path} {tmux} {env_name} {gpu_str}"],
        capture_output=True,
        text=True,
        shell=True,
        env=env,
    )

    # check for errors
    if out.returncode != 0:
        print("Error reported by tmux setup:")
        print(out.stdout)
        exit()

    # start new terminal and attach tmux session
    subprocess.call(
        [
            "gnome-terminal",
            "--",
            "bash",
            "-c",
            f"tmux a -t {tmux}",
        ]
    )


def find_tmux_fp():
    # baseline tmux file
    parent_dp = os.path.dirname(os.path.abspath(__file__))
    tmux_framework_fp = os.path.abspath(os.path.join(parent_dp, "tmux_baseline.sh"))

    return tmux_framework_fp


if __name__ == "__main__":
    import fire

    fire.Fire(start)
