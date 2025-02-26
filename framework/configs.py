from typing import Any
from pydantic import BaseModel, Field
import toml
import os
from typing import Union, Literal, List
import importlib
from datetime import datetime
import shutil
import json

import framework.util.file_utils as file_utils
import framework.util.train_util as train_utils


class RunConfig(BaseModel):
    gpu_id: Union[int, List[int]] = 0
    device_req_free: Union[bool, int] = True
    gpu_max_memory_fraction: float = 0.95
    max_train_steps: int = 100000
    save_every_n_epochs: int = 1
    train_n_workers: int = 0
    val_n_workers: int = 4
    num_sanity_val_steps: int = 1
    check_val_every_n_epoch: int = 1
    shuffle_dataset: Union[bool, int] = True
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high"
    deterministic: Union[bool, int] = False

    render_solid_background: Union[bool, int] = False
    run_name_postfix: str = ""
    experiment_category: str = ""

    resume_from_ckpoint: Union[bool, int] = False
    ckpoint_fp: str = None

    dataset_name: str = None
    dataset_limit_train_images: Union[int, bool] = False

    # this will be set by the main config during setup
    run_name: str = None

    # paths
    workspace_dp: str = None
    cache_dp: str = None
    datasets_dp: str = None
    run_dp: str = None

    @property
    def dataset_dp(self):
        return os.path.join(self.datasets_dp, self.dataset_name)

    def first_free_gpu(self):
        if not self.device_req_free:
            return self.gpu_id[0] if isinstance(self.gpu_id, list) else self.gpu_id
        free_gpus = train_utils.get_list_of_free_cuda_devices(allowed=self.gpu_id)
        return free_gpus[0]

    def sanity_checks(self):
        assert os.path.exists(self.workspace_dp), "workspace_dp not a valid directory"
        assert os.path.exists(self.cache_dp), "cache_dp not a valid directory"
        assert os.path.exists(self.dataset_dp), "dataset can not be found"
        # assert len(self.experiment_category) > 0, "experiment_category needs to be set to group trainings"


class MainConfig:
    def __init__(
        self, run_ifp, pipeline_ifp, legacy_path_ifp=None, legacy_location_config_ifp=None
    ) -> None:
        self.run = RunConfig(**toml.load(run_ifp))
        # loading pipeline config is more complicated since we have to figure out which cfg to load first
        pipeline_cfg_data = toml.load(pipeline_ifp)
        name = pipeline_cfg_data["pipeline"].split(".")
        # import the file where the named pipeline is in to find the init_config method
        pipeline_file = importlib.import_module(".".join(name[:-1]))
        self.pipeline = getattr(pipeline_file, name[-1]).init_config(pipeline_cfg_data)
        # handle legacy support for a separate location config
        if legacy_location_config_ifp is not None and os.path.exists(
            legacy_location_config_ifp
        ):
            self.run.dataset_name = toml.load(legacy_location_config_ifp)["aoi_id"]
        # handle legacy support for a separate paths config
        if legacy_location_config_ifp is not None and os.path.exists(
            legacy_location_config_ifp
        ):
            path_cfg = toml.load(legacy_location_config_ifp)
            for key, value in path_cfg.items():
                self.run[key] = value

        self.run.sanity_checks()
        if hasattr(self.pipeline, "sanity_checks") and callable(
            getattr(self.pipeline, "sanity_checks")
        ):
            self.pipeline.sanity_checks()

    def __str__(self) -> str:
        return f"RunConfig({str(self.run)}) \nPipelineConfig({str(self.pipeline)})"

    def create_run_name(self):
        run_name = (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
            + self.pipeline.pipeline.split(".")[-2]
            + f"_v{self.pipeline.version}_{self.run.dataset_name}"
        )
        if hasattr(self.pipeline, "determine_run_name_postfix") and callable(
            getattr(self.pipeline, "determine_run_name_postfix")
        ):
            pipeline_postfix = self.pipeline.determine_run_name_postfix()
            if pipeline_postfix is not None and len(pipeline_postfix) > 0:
                run_name += "_" + pipeline_postfix.strip("_")
        if len(self.run.run_name_postfix) > 0:
            run_name += "_" + self.run.run_name_postfix.strip("_")
        if self.run.resume_from_ckpoint:
            run_name += "_resume"

        self.set_existing_run_name(run_name)

    def set_existing_run_name(self, run_name):
        self.run.run_name = run_name
        workspace_dp = self.run.workspace_dp
        if (
            self.run.experiment_category is not None
            and len(self.run.experiment_category) > 0
        ):
            workspace_dp = os.path.join(
                workspace_dp, "_" + self.run.experiment_category.strip("_")
            )
        self.run.run_dp = os.path.join(workspace_dp, run_name)

    @staticmethod
    def get_instance():
        assert _config is not None
        return _config

    @staticmethod
    def set_instance(config):
        global _config
        _config = config

    def dump_to_toml(self, output_dp, pipeline_out_name="pipeline", run_out_name="run"):
        with open(os.path.join(output_dp, pipeline_out_name + ".toml"), "wt") as fp:
            toml.dump(json.loads(self.pipeline.json()), fp)
        with open(os.path.join(output_dp, run_out_name + ".toml"), "wt") as fp:
            toml.dump(json.loads(self.run.json()), fp)


def load_configs(
    run_config_fp: str,
    pipeline_config_fp: str,
    legacy_paths_config_fp: str = None,
    legacy_location_config_fp: str = None,
):

    if not os.path.exists(run_config_fp):
        os.makedirs(os.path.dirname(os.path.join(".", run_config_fp)), exist_ok=True)
        shutil.copy(
            os.path.join(".", "run", "run_template.toml"),
            os.path.join(".", run_config_fp),
        )
        if os.path.basename(run_config_fp) != "default.toml":
            print("Invalid run config path: ", run_config_fp)
        print("A new configuration file has been placed in the following location:")
        print(os.path.abspath(run_config_fp))
        print("Adapt the template values for your use case and run the script again")
        exit()

    file_utils.assert_fp_exists(run_config_fp)
    file_utils.assert_fp_exists(pipeline_config_fp)

    cfg = MainConfig(
        run_config_fp,
        pipeline_config_fp,
        legacy_paths_config_fp,
        legacy_location_config_fp,
    )
    MainConfig.set_instance(cfg)
    return cfg


def load_configs_from_logs(logs_dp: str):
    cfgs = load_configs(
        os.path.join(logs_dp, "configs", "run.toml"),
        os.path.join(logs_dp, "configs", "pipeline.toml"),
        os.path.join(logs_dp, "configs", "paths.toml"),
        os.path.join(logs_dp, "configs", "location.toml"),
    )

    return cfgs


def adapt_configs_for_inference(cfgs):
    """
    This method includes all config changes that need to be made when loading from disk
    but using the pipeline for inference instead of further training
    :param cfgs: configs
    :return: adapted configs
    """
    return cfgs
