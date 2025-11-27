from abc import ABC, abstractmethod
import DataLoader
import pandas as pd
import yaml
import sys
import logging
log = logging.getLogger(__name__)


class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = output_dir
        # load all config into member variables
        if not (config is None):
            for key, value in config.items():
                setattr(self, key, value)

    @abstractmethod
    def run(self, dl_dict):
        pass


    @abstractmethod
    def finalize(self):
        pass
