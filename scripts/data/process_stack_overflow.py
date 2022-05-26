import argparse
import json
import os
from typing import Dict, List

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.utils.run import make_deterministic


def main(conf: DictConfig, split_num: int = 1):
    make_deterministic(conf.seed)
    events = read_stackoverflow(to_absolute_path(conf.raw_data_dir))
    train_events, test_events = train_test_split(events, test_size=conf.test_size)
    train_events, val_events = train_test_split(
        train_events, test_size=1 - (conf.train_size / (conf.train_size + conf.val_size))
    )

    output_data_path = f"{to_absolute_path(conf.save_dir)}/stack_overflow/split_{split_num}"
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    for split_name, split in zip(
        ["train", "test", "val"], [train_events, test_events, val_events]
    ):
        with open(f"{output_data_path}/{split_name}.json", "w") as f:
            json.dump(split, f)


def read_stackoverflow(data_path: str) -> List[List[Dict]]:
    time_seqs: List[List[float]] = []
    with open(f"{data_path}/time.txt") as ftime:
        seqs = ftime.readlines()
        for seq in seqs:
            time_seqs.append([float(t) for t in seq.split()])

    mark_seqs: List[List[int]] = []
    with open(f"{data_path}/event.txt") as fmark:
        seqs = fmark.readlines()
        for seq in seqs:
            mark_seqs.append([int(float(k)) for k in seq.split()])

    events: List[List[Dict]] = []
    for times, marks in zip(time_seqs, mark_seqs):
        events.append([{"time": time, "labels": [label - 1]} for time, label in zip(times, marks)])
    return events


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", type=str, default="./data/", help="A directory to save processed datset"
)
parser.add_argument(
    "--raw-data-dir", type=str, default="./data/raw/stack_overflow", help="A directory of raw data"
)
parser.add_argument(
    "--splits-num", type=int, default=1, help="How many different splits to generate."
)
parser.add_argument("--experiment", type=str, default="stack_overflow", help="Experiment name")

if __name__ == "__main__":
    parser_args = parser.parse_args()
    config = OmegaConf.load(f"./config/data/{parser_args.experiment}.yaml")
    config["raw_data_dir"] = parser_args.raw_data_dir
    config["save_dir"] = parser_args.data_dir
    for split_num in range(parser_args.splits_num):
        config["seed"] = split_num * 42
        main(config, split_num + 1)
