import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tick.hawkes import HawkesKernelExp, SimuHawkes
from tpp.utils.record import hawkes_seq_to_record
from tqdm import tqdm

from src.utils.run import make_deterministic


def generate_points(
    n_processes: int,
    mu: np.array,
    alpha: np.array,
    decay: np.array,
    window: int,
    seed: int,
    dt=0.01,
):
    """
    Generates points of an marked Hawkes processes using the tick library
    """
    hawkes = SimuHawkes(n_nodes=n_processes, end_time=window, verbose=False, seed=seed)
    for i in range(n_processes):
        for j in range(n_processes):
            hawkes.set_kernel(
                i=i,
                j=j,
                kernel=HawkesKernelExp(intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]),
            )
        hawkes.set_baseline(i, mu[i])

    hawkes.track_intensity(dt)
    hawkes.simulate()
    return hawkes.timestamps


def parse_hawkes_parameters(conf: DictConfig) -> Tuple[np.array, np.array, np.array]:
    """
    Parses Hawkes parameters from Hydra DictConfig.

    returns mu, alpha, beta
    """
    mu = np.array(conf.mu).astype(np.float64)
    alpha = np.array(conf.alpha).astype(np.float64).reshape(mu.shape * 2)
    beta = np.array(conf.beta).astype(np.float64).reshape(mu.shape * 2)

    return mu, alpha, beta


def main(conf: DictConfig, split_num: int = 1) -> None:
    """
    Uses Hawkes to generate data based on the
    given params.
    """

    hawkes_data_dir = os.path.join(
        os.path.expanduser(conf.data_dir), "hawkes", f"split_{split_num}"
    )
    Path(hawkes_data_dir).mkdir(parents=True, exist_ok=True)

    seeds = {
        "train": [conf.train_size, conf.seed],
        "val": [conf.val_size, conf.seed + conf.train_size],
        "test": [conf.test_size, conf.seed + conf.train_size + conf.val_size],
    }

    mu, alpha, beta = parse_hawkes_parameters(conf)

    for name, [size, seed] in seeds.items():
        range_size = range(size)
        if conf.verbose:
            range_size = tqdm(range_size)

        times_marked = [
            generate_points(
                n_processes=conf.marks,
                mu=mu,
                alpha=alpha,
                decay=beta,
                window=conf.window,
                seed=seed + i,
            )
            for i in range_size
        ]

        with open(os.path.join(hawkes_data_dir, name + ".json"), "w") as h:
            h.write(
                "["
                + ",\n".join(
                    json.dumps(i) for i in [hawkes_seq_to_record(r) for r in times_marked]
                )
                + "]\n"
            )


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./data/", help="A base dataset directory")
parser.add_argument(
    "--splits-num", type=int, default=1, help="How many different splits to generate."
)
parser.add_argument("--experiment", type=str, default="hawkes", help="Experiment name")

if __name__ == "__main__":
    parser_args = parser.parse_args()
    config = OmegaConf.load(f"./config/data/{parser_args.experiment}.yaml")
    config["data_dir"] = parser_args.data_dir
    for split_num in range(parser_args.splits_num):
        config["seed"] = split_num * 42
        make_deterministic(seed=config.seed)
        main(config, split_num + 1)
