from abc import ABC, abstractmethod
import DataLoader
import pandas as pd
import yaml
import sys
import logging
log = logging.getLogger(__name__)


class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config):
        self.m_config = config

    @abstractmethod
    def run(self, dl: DataLoader.DataLoader):
        pass


    @abstractmethod
    def finalize(self):
        pass
