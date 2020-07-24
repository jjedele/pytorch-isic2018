import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from data import Ham10kDataModule


class Ham10kModel(pl.LightningModule):
    def __init__(self, data_module):
        super(Ham10kModel, self).__init__()
        self.data_module = data_module
        self.net = models.resnet18(
            pretrained=True,
            progress=True,
            #num_classes=7,#len(train_dataset.class_weights),
        )
        self.net.fc = torch.nn.Linear(512, 7)
        self.metric = Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = self.metric(y_pred, y)
        logs = {"accuracy": acc}
        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = self.metric(y_pred, y)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        data_obj = {"val_loss": avg_loss, "val_acc": avg_acc}
        data_obj["log"] = {"val_loss": avg_loss, "val_acc": avg_acc}
        print("VAL", data_obj)
        return data_obj

    def configure_optimizers(self):
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
        return optimizer

    def prepare_data(self):
        return self.data_module.prepare_data()

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()


if __name__ == "__main__":
    data_module = Ham10kDataModule()
    model = Ham10kModel(data_module)

    trainer = pl.Trainer()
    trainer.fit(model)
