import importlib
import json
import sys
import os

from data_prep.utils.dataset_config import (
    DatasetConfig,
    create_config_from_template,
)


def run_processing_step(step_cfg, cfg, state):
    processing_step_fp = step_cfg.file
    print()
    print("=======================================================================")
    print("Processing step:", processing_step_fp)
    print("=======================================================================")

    # allow importing processing steps from other directories
    if step_cfg.from_dir is not None:
        sys.path.append(os.path.abspath(step_cfg.from_dir))

    adapter_file = importlib.import_module(processing_step_fp)
    adapter = getattr(adapter_file, "ProcessingStep")(cfg, step_cfg, state)

    if step_cfg.enabled:
        if cfg.general.lazy and adapter.can_be_skipped(cfg, state):
            print("Skipping because of lazy")
        else:
            adapter.run(cfg, state)

    adapter.update_state(cfg, state, step_cfg.enabled)

    print("=======================================================================")
    print("Finished Processing step", processing_step_fp)
    print("Updated State: ")
    print(json.dumps(state, indent=2))
    print("=======================================================================")
    print()


def create_dataset(cfg):
    state = {}

    for step in cfg.steps:
        run_processing_step(
            step,
            cfg,
            state,
        )


def run_create_dataset(
    cfg_ifp="./configs/data/dataset.cfg",
    cfg_template_fp="./data_prep/utils/dataset_template.cfg",
):
    cfg = create_config_from_template(
        cfg_ifp, cfg_template_fp, DatasetConfig.get_from_file
    )
    DatasetConfig.set_instance(cfg)
    create_dataset(cfg)


if __name__ == "__main__":
    import fire

    fire.Fire(run_create_dataset)
