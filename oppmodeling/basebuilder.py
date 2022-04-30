from argparse import ArgumentParser
from glob import glob
from os import makedirs, listdir
from os.path import join, split, basename, exists
from tqdm import tqdm
import shutil
import time

import torch

import os
BASE_STORAGE = os.environ['OppModelingStorage']

def add_builder_specific_args(parser, datasets):
    for ds in datasets:
        if ds.lower() == "combined":
            from oppmodeling.dataset_builders import CombinedBuilder

            parser = CombinedBuilder.add_data_specific_args(parser, name=ds)
        else:
            raise NotImplementedError(f"{ds} not implemented")
    return parser


def create_builders(hparams):
    """
    Used in DataModule which leverages several different datasets.
    """
    builders = []
    for ds in hparams["datasets"]:
        if ds.lower() == "combined":
            from oppmodeling.dataset_builders import CombinedBuilder

            tmp_builder = CombinedBuilder(hparams)
        else:
            raise NotImplementedError(f"{ds} not implemented")
        builders.append(tmp_builder)
    return builders


# Superclass used for all datasets
class BaseBuilder(object):
    def __init__(self, hparams):
        if not isinstance(hparams, dict):
            hparams = vars(hparams)

        self.hparams = hparams
        self.set_paths()

    def set_paths(self):
        self.root = self.hparams[f"root_{self.NAME.lower()}"]
        makedirs(self.root, exist_ok=True)

    def prepare_data(self, tokenizer):
        
        #first, process the data -> implemented by child class
        self._process_data(tokenizer)

        #must be filled by now.
        assert self.processed_data
        assert 'train' in self.processed_data and 'val' in self.processed_data and 'test' in self.processed_data

        #we are done; data is processed and is in the right format.

    def _process_data(self, tokenizer):
        """
        TO BE IMPLEMENTED BY EACH DATASET.
        Each data will fill in self.processed_data with the processed data
        """
        self.processed_data = None

    @staticmethod
    def add_data_specific_args(parent_parser, name="name"):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )
        parser.add_argument(f"--root_{name}", type=str, default=f"{BASE_STORAGE}/data/{name}")

        return parser