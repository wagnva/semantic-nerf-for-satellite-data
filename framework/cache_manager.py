import os


class CacheDir:
    def __init__(self, cfgs, name=None) -> None:
        super().__init__()
        self.cfgs = cfgs
        name = cfgs.run.dataset_name if name is None else name
        self.cache_dp = os.path.join(cfgs.run.cache_dp, name)

    def exists(self, name) -> bool:
        """
        This method checks if for the given run, a cache with the given name has been already created
        This only checks if at least one file has been saved in the named cache dir
        Does not check for validity of saved information
        :param name: name of the cache dir
        :return:
        """
        dir_path = os.path.join(self.cache_dp, name)
        return (
            os.path.exists(dir_path)
            and os.path.isdir(dir_path)
            and len(os.listdir(dir_path)) > 0
        )

    def dir_path(self, name):
        """
        Returns the path to a cache dir with a given name
        :param name: name of the cache dir
        :return: path to the cache dir for the name in a given run
        """
        dir_path = os.path.join(self.cache_dp, name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
