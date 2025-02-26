from enum import Enum
import toml
import os
from pydantic import BaseModel
from typing import List, Union  # , Literal
from typing_extensions import Literal
from shutil import copyfile


_config = None


class GeneralConfig(BaseModel):
    lazy: Union[bool, int] = 0
    name_appendix: str = None
    workspace_dp: str = None

    def sanity_checks(self):
        assert os.path.exists(self.workspace_dp), "Necessary path doesnt exist"


class Step(BaseModel):
    file: str = None
    enabled: Union[bool, int] = True
    from_dir: str = None
    data: dict = None

    def get(self, data_key, default=None):
        if self.data is None:
            return None
        return self.data.get(data_key, default)

    def sanity_checks(self):
        assert len(self.file) > 0, "path to processing step is not set"
        if self.from_dir is not None:
            assert os.path.exists(
                self.from_dir
            ), "trying to import processing_step from non-existing directory"
        # if "DFC2019" in self.file:
        #     assert os.path.exists(self.dfc2019_truth_dp), "Necessary path doesnt exist"
        #     assert os.path.exists(self.dfc2019_rgb_dp), "Necessary path doesnt exist"
        #     assert os.path.exists(self.dfc2019_metadata_dp), "Necessary path doesnt exist"


class SiteConfig(BaseModel):
    location_name: str = None
    zone_string: str = None
    # if set, an alternative region-of-interest is used
    # normally the roi is the DSM Ground Truth Region
    alternative_roi_fp: str = None
    # if not set, use min/max values of the GT-DSM
    alt_min: float = None
    alt_max: float = None


class TrainTestConfig(BaseModel):
    max_samples: int = -1
    train_test_file_split_method: Literal[
        "use_predefined_test_files",
        "use_custom_test_files",
        "use_fixed_test_file_amount",
        "random_test_files",
    ] = "use_predefined_test_files"
    custom_test_files: List = None
    fixed_test_file_amount: int = None
    # shuffle dataset before train/test split
    shuffle_dataset: Union[bool, int] = False
    # define a subset of files that are used for train/test split
    subset_files: List = None
    # exclude specific files
    exclude_files: List = None

    def sanity_checks(self):
        if self.train_test_file_split_method == "use_custom_test_files":
            assert self.custom_test_files is not None, "custom_test_files not set"
        if self.train_test_file_split_method == "use_fixed_test_file_amount":
            assert (
                self.fixed_test_file_amount is not None
            ), "fixed_test_file_amount not set"


class DatasetConfig(BaseModel):
    general: GeneralConfig = GeneralConfig()
    site: SiteConfig = SiteConfig()
    files: TrainTestConfig = TrainTestConfig()
    steps: List[Step] = []

    def sanity_checks(self):
        self.general.sanity_checks()
        self.files.sanity_checks()
        for step in self.steps:
            step.sanity_checks()

    @property
    def aoi_id(self):
        max_samples = self.files.max_samples
        suffix = ""
        if max_samples > 0:
            suffix = f"_len_{max_samples}"
        if self.general.name_appendix is not None and len(self.general.name_appendix) > 0:
            suffix += f"_{self.general.name_appendix}"
        return self.site.location_name + suffix

    @property
    def output_dp(self):
        return os.path.join(self.general.workspace_dp, self.aoi_id)

    @staticmethod
    def get_instance():
        assert _config is not None
        return _config

    @staticmethod
    def set_instance(config):
        global _config
        _config = config

    @classmethod
    def get_from_file(cls, toml_ifp):
        config_dict = toml.load(toml_ifp)
        cfg = cls(**config_dict)
        cfg.sanity_checks()
        return cfg


def create_config_from_template(config_fp, config_template_ifp, inst_cfg_fn):
    """
    Makes sure that a .cfg exists at <config_fp>. If not, copies over the template and stops the script
    to give the user the chance for configuration
    If the .cfg exists at <config_fp>., the <inst_cfg_fn> function is called with the correct path
    :param config_fp: path to the (possible) actual config file
    :param config_template_ifp: path to the template config file
    :param inst_cfg_fn: function to generate a config based on the content of the .cfg file
    :return: the result of inst_cfg_fn, which should be a configuration data class
    """
    config_template_ifp = os.path.abspath(config_template_ifp)
    config_fp = os.path.abspath(config_fp)

    if not os.path.isfile(config_fp):
        os.makedirs(os.path.dirname(config_fp), exist_ok=True)
        copyfile(config_template_ifp, config_fp)
        print("A new configuration file has been placed in the following location:")
        print(config_fp)
        print("Adapt the template values for your use case and run the script again")
        exit()

    return inst_cfg_fn(config_fp)
