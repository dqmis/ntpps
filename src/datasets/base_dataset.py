import json
import os
from abc import ABC, abstractstaticmethod
from typing import List

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractstaticmethod
    def to_features(batch):
        raise NotImplementedError()

    def load_data(self, data_dir: str, load_from_dir: str, name: str) -> List:
        """
        Loads dataset. If load_dir provided, dataset is loaded.
        """
        try:
            data_dir = os.path.join(data_dir, load_from_dir)
            data_path = os.path.join(data_dir, name + ".json")
            with open(data_path, "r") as h:
                records = json.load(h)
            return records
        except:
            raise Exception("Wrong data dir is provided")
