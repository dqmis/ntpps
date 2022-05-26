import inspect
from typing import Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.datasets.diffeq_dataset import DiffEqDataset
from src.datasets.unmarked_multi_class_dataset import UnmarkedMultiClassDataset

DATASET_TYPES = {"diffeq": DiffEqDataset, "ntpp": UnmarkedMultiClassDataset}


def parse_dataset_arguments(conf: DictConfig):
    supported_params = list(inspect.signature(DATASET_TYPES[conf.model_type].__init__).parameters)
    return {k: conf[k] for k in supported_params if k in conf}


def config_dataset(conf: DictConfig, name: str) -> BaseDataset:
    if not conf.model_type in DATASET_TYPES.keys():
        raise Exception(f"Dataset for model {conf.model_type} was not found!")

    parsed_params = parse_dataset_arguments(conf)
    parsed_params["name"] = name
    return DATASET_TYPES[conf.model_type](**parsed_params)


def get_loaders(conf: DictConfig):
    """
    Returns train, val, test loaders
    """
    loaders: Tuple = ()
    for name, shuffle in zip(["train", "val", "test"], [True, False, False]):
        loaders += (
            DataLoader(
                config_dataset(conf, name),
                batch_size=conf.batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=DATASET_TYPES[conf.model_type].to_features,
            ),
        )
    return loaders[0], loaders[1], loaders[2]
