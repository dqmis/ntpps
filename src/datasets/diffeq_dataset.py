from torchdiffeq.utils import create_tsave

from src.datasets.base_dataset import BaseDataset


class DiffEqDataset(BaseDataset):
    def __init__(self, scale: float, dt: float, name: str, data_dir: str, load_from_dir: str):
        self._scale = scale
        self._dt = dt
        self._events = self.load_data(data_dir, load_from_dir, name)

        self._max_event_time = (
            max(max(y["time"] for y in x) for x in self._events)
            - min(min(y["time"] for y in x) for x in self._events)
        ) * self._scale
        self._min_time_event = min(min(y["time"] for y in x) for x in self._events)

    def __len__(self):
        return len(self._events)

    def __getitem__(self, idx):
        return {
            "max_event_time": self._max_event_time,
            "dt": self._dt,
            "events": [
                ((x["time"] - self._min_time_event) * self._scale, x["labels"][0])
                for x in self._events[idx]
            ],
        }

    @staticmethod
    def to_features(batch):
        max_event_time = batch[0]["max_event_time"]
        dt = batch[0]["dt"]
        batch_size = len(batch)
        batch = [x["events"] for x in batch]

        evnts_raw = sorted(
            (evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]
        )
        tsave, _, evnts, tse = create_tsave(0, max_event_time + dt, dt, evnts_raw)
        return {
            "tsave": tsave,
            "evnts": evnts,
            "tse": tse,
            "batch_size": batch_size,
        }
