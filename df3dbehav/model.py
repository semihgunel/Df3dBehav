import cv2
import pandas as pd
import pickle5
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import sys
import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
import numpy as np

class ClassifierDf3d(pl.LightningModule):
    def __init__(self, labels=None, n_classes:int=7, background_index:int=7):
        super().__init__()
        self.labels = labels
        self.encoder = nn.Sequential(
            Rearrange('b t c -> b c t'),
            nn.Conv1d(76, 38, kernel_size=5, padding=2),
            nn.BatchNorm1d(38),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Conv1d(38, 38, kernel_size=5, padding=2),
            nn.BatchNorm1d(38),
            nn.ReLU(),
            nn.Conv1d(38, 38, kernel_size=5, padding=2),
            nn.BatchNorm1d(38),
            nn.ReLU(),
            nn.Conv1d(38, 38, kernel_size=5, padding=2),
            nn.BatchNorm1d(38),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Conv1d(38, 19, kernel_size=5, padding=2),
            nn.BatchNorm1d(19),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Conv1d(19, 9, kernel_size=5, padding=2),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Conv1d(9, n_classes, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_classes),
            nn.ReLU(),
            nn.Conv1d(n_classes, n_classes, kernel_size=5, padding=2),
            )
        self.background_index = background_index
        self.loss = nn.CrossEntropyLoss(ignore_index=background_index)
    
    def get_accuracy(self, y_hat, y, b):
        nb_indices = (y == b[:, None]) # (b x time), find non-background indices
        pred = (y_hat * nb_indices[:,None,:]).sum(-1) # (b x n_classes) summing predictions over non-background indices
        pred = pred.softmax(-1)
        pred_b = torch.argmax(pred, dim=-1)
        acc = (pred_b == b).sum() / pred.size(0) # (b)
        return acc, pred_b
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def plot_confusion_matrix(self, y, y_hat, labels, n):
        fig = plt.figure()
        plot_confusion_tensorboard(
            np.concatenate([y.cpu().data.numpy(), np.arange(7)]),
            np.concatenate([y_hat.cpu().data.numpy(), np.arange(7)]),
            labels=labels,
            ax=plt.gca(),
            normalize=None
        )
        self.logger.experiment.add_figure(
            tag=f"{n}/conf", figure=fig, global_step=self.current_epoch, close=True,
        )

    def training_step(self, train_batch, batch_idx):
        x, y, meta = train_batch
        y_hat = self.encoder(x)
        loss = self.loss(y_hat, y)
        acc, pred_b = self.get_accuracy(y_hat, y, meta['behav'].cuda())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True, )
        return loss
    
    def validation_step(self, train_batch, batch_idx):
        x, y, meta = train_batch
        y_hat = self.encoder(x)
        loss = self.loss(y_hat, y)
        acc, pred_b = self.get_accuracy(y_hat, y, meta['behav'].cuda())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if batch_idx == 0:
            self.plot_confusion_matrix(meta['behav'], pred_b, self.labels, "val")
        return loss