from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from tpp.models import get_model
from tpp.models.base.process import Process as BaseProcess
from tpp.utils.events import get_events, get_window
from tpp.utils.lr_scheduler import create_lr_scheduler


class NeuralTPP(pl.LightningModule):
    def __init__(
        self,
        marks: int,
        multi_labels: bool,
        device: str,
        encoder_decoder_args: Dict,
        optimizer_args: Dict,
        lr_poisson_rate_init: float,
        lr_rate_init: float,
        time_scale: float,
        use_coefficients: bool,
        window: Optional[int] = None,
        padding_id: int = -1,
        include_poisson: bool = False,
    ):
        super().__init__()
        self._time_scale = time_scale
        self._padding_id = padding_id
        self._lr_rate_init = lr_rate_init
        self._lr_poisson_rate_init = lr_poisson_rate_init
        self._include_poisson = include_poisson
        self._window = window
        self._device = self._init_device(device)
        self._multi_labels = multi_labels
        self._marks = marks
        self._optimizer_args = optimizer_args

        self._process: BaseProcess = get_model(
            encoder_decoder_args,
            marks,
            multi_labels,
            include_poisson,
            use_coefficients,
            device=self._device,
        )

    def _init_device(self, device: str):
        return torch.device(device)

    def forward(self, x, inference: bool = False):
        times = x["times"]
        additional_features = x["additional_features"]

        mask = (times != -1).type(times.dtype)
        window_start, window_end = get_window(times=times, window=self._window)
        events = get_events(
            times=times,
            mask=mask,
            labels=x["labels"],
            window_start=window_start,
            window_end=window_end,
        )
        events_times = events.get_times()

        log_p, y_pred_mask = self._process.log_density(query=events_times, events=events, af=additional_features)
        if self._multi_labels:
            y_pred = log_p
        elif inference and not self._multi_labels:
            y_pred = log_p
        else:
            y_pred = log_p.argmax(-1).type(log_p.dtype)
        return y_pred, y_pred_mask

    def predict(self, dataloader):
        def detach(x: torch.Tensor):
            return x.cpu().detach().numpy()

        self._process.eval()

        predictions: List[List[Tuple]] = []
        for batch in dataloader:

            times, labels = batch["times"], batch["labels"]
            labels = (labels != 0).type(labels.dtype)

            mask = (times != self._padding_id).type(times.dtype)
            window_start, window_end = get_window(times=times, window=self._window)

            events = get_events(
                times=times,
                mask=mask,
                labels=labels,
                window_start=window_start,
                window_end=window_end,
            )

            y_pred, _ = self.forward(batch, inference=True)
            if self._multi_labels:
                labels = events.labels
            else:
                labels = events.labels.argmax(-1).type(events.labels.dtype)
            times = detach(times)
            labels = detach(labels)
            y_pred = detach(y_pred)
            batch_predictions: List[Tuple[float, int, np.array]] = []
            for batch_idx, _ in enumerate(times):
                unpadded_time_points = times[batch_idx] != -1
                unpadded_times = times[batch_idx][unpadded_time_points]
                unpadded_labels = labels[batch_idx][unpadded_time_points]
                unpadded_predictions = y_pred[batch_idx][unpadded_time_points]
                batch_predictions.append(
                    [
                        (time, int(true_label), predictions)
                        for time, true_label, predictions in zip(
                            unpadded_times, unpadded_labels, unpadded_predictions
                        )
                    ]
                )
            predictions.extend(batch_predictions)
        return predictions

    def configure_optimizers(self):
        if self._include_poisson:
            processes = self._process.processes.keys()
            modules = []
            for p in processes:
                if p != "poisson":
                    modules.append(getattr(self._process, p))
            optimizer = Adam(
                [{"params": m.parameters()} for m in modules]
                + [{"params": self._process.alpha}]
                + [
                    {
                        "params": self._process.poisson.parameters(),
                        "lr": self._lr_poisson_rate_init,
                    }
                ],
                lr=self._lr_rate_init,
            )
        else:
            optimizer = Adam(self._process.parameters(), lr=self._lr_rate_init)

        self._optimizer_args["optimizer"] = optimizer
        scheduler = create_lr_scheduler(**self._optimizer_args)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        times, labels, additional_features = (
            batch["times"],
            batch["labels"],
            batch["additional_features"],
        )
        labels = (labels != 0).type(labels.dtype)

        seq_lens = batch["seq_lens"]
        max_seq_len = seq_lens.max()
        times, labels = times[:, :max_seq_len], labels[:, :max_seq_len]

        mask = (times != self._padding_id).type(times.dtype)
        times = times * self._time_scale

        window_start, window_end = get_window(times=times, window=self._window)
        events = get_events(
            times=times, mask=mask, labels=labels, window_start=window_start, window_end=window_end
        )

        loss, loss_mask, _ = self._process.neg_log_likelihood(
            events=events, af=additional_features
        )
        loss = torch.sum(loss * loss_mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        times, labels, additional_features = (
            batch["times"],
            batch["labels"],
            batch["additional_features"],
        )
        labels = (labels != 0).type(labels.dtype)

        seq_lens = batch["seq_lens"]
        max_seq_len = seq_lens.max()
        times, labels = times[:, :max_seq_len], labels[:, :max_seq_len]

        mask = (times != self._padding_id).type(times.dtype)
        times = times * self._time_scale

        window_start, window_end = get_window(times=times, window=self._window)
        events = get_events(
            times=times, mask=mask, labels=labels, window_start=window_start, window_end=window_end
        )

        loss, loss_mask, _ = self._process.neg_log_likelihood(
            events=events, af=additional_features
        )
        loss = torch.sum(loss * loss_mask)
        self.log("val_loss", loss)
