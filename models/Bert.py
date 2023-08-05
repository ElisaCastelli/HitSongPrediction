import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split
import os
from torchmetrics.regression import R2Score
from TracksDataset import TracksDataset
from PreTrainedResNet import PreTrainedResnet

neptune_logger = NeptuneLogger(
    project="elisacastelli/tesi",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTE3NDY5My02ZjBkLTQxMjktYjY0ZS0wMGI1NmU0NzNjNmYifQ==",
    log_model_checkpoints=True,
)

class BERTPopularity(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')