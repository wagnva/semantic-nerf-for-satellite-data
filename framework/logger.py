import logging
from logging.config import fileConfig
import os
import time


class Logger:
    def __init__(self) -> None:
        super().__init__()
        self.topic_lvl = 0
        self._min_length_prefix = 12
        # self.formatter = logging.Formatter('%(asctime)s %(message)s', '%H:%M')
        self.formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self._logger.addHandler(ch)
        self._counter = {}

    def info(self, tag: str = "Main", msg=""):
        self._logger.info(self.format(tag, msg))

    def debug(self, tag: str = "Main", msg="", every=-1):
        if self._every(tag, every):
            self._logger.debug(self.format(tag, msg, mode="D"))

    def error(self, tag: str = "Main", msg=""):
        self._logger.error(self.format(tag, msg, mode="E"))

    def format(self, tag: str = "Main", msg="", mode: str = ""):
        return (
            "    " * self.topic_lvl
            + f"[{mode}{'' if mode == '' else ':'}{tag}]".ljust(self._min_length_prefix)
            + f" {time.strftime('%H:%M')}: "
            + str(msg)
        )

    def _every(self, tag: str, every: int):
        if every < 0:
            return True
        counter = self._counter.get(tag)
        if counter is None:
            self._counter[tag] = 0
            counter = 0

        self._counter[tag] += 1

        if counter > every:
            self._counter[tag] = 0
            return True
        return False

    def init_write_to_file(self, log_fp: str):
        fh = logging.FileHandler(log_fp)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self._logger.addHandler(fh)

    def init_tensorboard(self, tensorboard):
        self.tensorboard = tensorboard

    def subtopic(self):
        self.topic_lvl += min(1, 5)

    def supertopic(self):
        self.topic_lvl = max(self.topic_lvl - 1, 0)

    def reset_topic(self):
        self.topic_lvl = 0


logger = Logger()
