import json
import os
from audioop import add
from typing import Optional
from urllib.response import addinfo

import numpy as np
import pandas as pd
import torch
from tpp.utils.marked_times import objects_from_events, pad

from src.datasets.base_dataset import BaseDataset


class UnmarkedMultiClassDataset(BaseDataset):
    """
    MultiClassDataset: Unmarked Multi-class Hawkes dataset.
    The intensity function for this process is:
    lambda_i(t|tjk) = mu(i) + sum[
        alpha(i, j) * sum[exp(t - tjk) for tjk < t] for j in range(n_nodes)
    ]
    where tjk are timestamps of all events of node j
    Args
        device: device where the generated data is loaded
        n_processes: number of generated Hawkes processes
        padding_id: id of the padded value used to generate the dataset
    """

    def __init__(
        self,
        data_dir: str,
        load_from_dir: str,
        marks: int,
        name: str,
        device: str,
        use_additional_features: Optional[bool] = False,
    ):
        super(UnmarkedMultiClassDataset, self).__init__()
        self.data_dir = data_dir
        self.device = torch.device(device)
        self.n_processes = marks
        self.name = name
        self.padding_id = -1
        self.load_from_dir = load_from_dir
        self.times_dtype = torch.float32
        self.labels_dtype = torch.float32

        self.raw_objects = self._build_sequences()
        self.times = self.raw_objects["times"]
        self.labels = self.raw_objects["labels"]

        self.lengths = [len(x) for x in self.times]

        self.max_length = max(self.lengths)
        self.lengths = torch.Tensor(self.lengths).long().to(self.device)
        if use_additional_features:
            self.additional_features = self.load_additional_features()
        else:
            self.additional_features = None

    def load_additional_features(self) -> np.array:
        try:
            data_dir = os.path.join(self.data_dir, self.load_from_dir)
            data_path = os.path.join(data_dir, f"user_af_{self.name}.json")
            with open(data_path, "r") as h:
                additional_features = pd.DataFrame(json.load(h))
            return np.array(additional_features.values)
        except:
            raise Exception("Cannot load additional features")

    def _build_sequences(self):
        events = self.load_data(self.data_dir, self.load_from_dir, self.name)
        if "events" in events[0]:
            records = events
            events = [r["events"] for r in records]
            events = [e for e in events if len(e) > 0]

        # times, labels
        raw_objects = objects_from_events(
            events=events,
            marks=self.n_processes,
            labels_dtype=self.labels_dtype,
            device=self.device,
        )

        not_empty = [len(x) > 0 for x in raw_objects["times"]]

        def keep_not_empty(x):
            return [y for y, nonempty in zip(x, not_empty) if nonempty]

        return {k: keep_not_empty(v) for k, v in raw_objects.items()}

    def __getitem__(self, item):
        raw_objects = {k: v[item] for k, v in self.raw_objects.items()}
        if self.additional_features is None:
            additional_features = None
        else:
            additional_features = (
                torch.tensor(self.additional_features[item]).to(self.device).long()
            )
        seq_len = self.lengths[item]
        result = {
            "raw": raw_objects,
            "seq_len": seq_len,
            "padding_id": self.padding_id,
            "additional_features": additional_features,
        }
        return result

    def __len__(self):
        return len(self.times)

    @staticmethod
    def to_features(batch):
        """
        Casts times and events to PyTorch tensors
        """
        times = [b["raw"]["times"] for b in batch]
        if batch[0]["additional_features"] is None:
            additional_features = None
        else:
            additional_features = torch.stack([b["additional_features"] for b in batch])
        labels = [b["raw"]["labels"] for b in batch]
        padding_id = batch[0]["padding_id"]

        assert padding_id not in torch.cat(times)
        padded_times = pad(x=times, value=padding_id)  # [B,L]
        # Pad with zero, not with padding_id so that the embeddings don't fail.
        padded_labels = pad(x=labels, value=0)  # [B,L]

        features = {
            "times": padded_times,
            "labels": padded_labels,
            "additional_features": additional_features,
            "seq_lens": torch.stack([b["seq_len"] for b in batch]),
        }
        return features
