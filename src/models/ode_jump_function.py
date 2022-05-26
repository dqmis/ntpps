from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchdiffeq._impl import odeint_adjoint
from torchdiffeq.modules import ODEJumpFunc


class ODEJumpFunction(pl.LightningModule):
    def __init__(
        self,
        dim_c: int,
        dim_h: int,
        dim_N: int,
        jump_type: str,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        adam_lr: float = 1.0e-2,
        dim_hidden: int = 32,
        num_hidden: int = 2,
        ortho: bool = True,
    ):
        super().__init__()
        self.type_forecast = [0.0]
        self.rtol = 1.0e-5
        self.atol = 1.0e-7
        self.lr = lr
        self.adam_lr = adam_lr
        self.weight_decay = weight_decay
        self.c0 = torch.randn(dim_c, requires_grad=True)
        self.h0 = torch.zeros(dim_h)
        self.model = ODEJumpFunc(
            dim_c,
            dim_h,
            dim_N,
            dim_N,
            dim_hidden=dim_hidden,
            num_hidden=num_hidden,
            ortho=ortho,
            jump_type=jump_type,
            activation=nn.CELU(),
        )

    @property
    def jump_type(self) -> str:
        return self.jump_type

    def configure_optimizers(self):
        return Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.c0, "lr": self.adam_lr},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def forward(self, z):
        return self.model.forward(z)

    def training_step(self, batch, batch_idx):
        _, _, _, _, loss = self.forward_pass(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, _, loss = self.forward_pass(batch)
        self.log("val_loss", loss)
        return loss

    def predict(self, dataloader):
        predictions: List[List[Tuple]] = []
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                tsave, _, lmbda, tse, _ = self.forward_pass(batch)
                events = [[] for i in range(lmbda.shape[1])]
                for event_tgid_idx, event_batch_idx, event_cls in tse:
                    lmbds = lmbda[event_tgid_idx][event_batch_idx]
                    event_entry = (tsave[event_tgid_idx].item(), event_cls, lmbds.numpy())
                    events[event_batch_idx].append(event_entry)
                predictions.extend(events)
        return predictions

    def forward_pass(self, batch):
        z0 = torch.cat((self.c0, self.h0), dim=-1)
        tsave = batch["tsave"]
        evnts = batch["evnts"]
        tse = batch["tse"]
        batch_size = batch["batch_size"]
        self.model.evnts = evnts

        trace = odeint_adjoint(
            self.model,
            z0.repeat(batch_size, 1),
            tsave,
            method="jump_adams",
            rtol=self.rtol,
            atol=self.atol,
        )
        params = self.model.L(trace)
        lmbda = params[..., : self.model.dim_N]

        def integrate(tt, ll):
            lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
            dts = (tt[1:] - tt[:-1]).reshape((-1,) + (1,) * (len(lm.shape) - 1)).float()
            return (lm * dts).sum()

        log_likelihood = -integrate(tsave, lmbda)

        seqs_happened = {sid for sid in range(batch_size)}

        et_error = []
        for evnt in tse:
            log_likelihood += torch.log(lmbda[evnt])
            if evnt[1] in seqs_happened:
                type_preds = torch.zeros(len(self.type_forecast))
                for tid, t in enumerate(self.type_forecast):
                    loc = (np.searchsorted(tsave.numpy(), tsave[evnt[0]].item() - t),) + evnt[1:-1]
                    type_preds[tid] = lmbda[loc].argmax().item()
                et_error.append((type_preds != evnt[-1]).float())
            seqs_happened.add(evnt[1])

        return tsave, trace, lmbda, tse, -log_likelihood
