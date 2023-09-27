import torch.nn as nn
import torch
import torchvision.models as models
from torchmetrics import Accuracy
from GTZANDataloader import GTZANDataModule
from FMADataLoader import FMADataloader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.models import ResNet50_Weights

neptune_logger = NeptuneLogger(
    project="elishcastle/HSP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTkxYWQyNi1kMWM3LTRiNzYtYWJhNS05ZThmNWZiNmMwOTIifQ==",
)
# TODO FAI PARTIRE UNO COME HSP-53 MA TENENDO I DROPOUT COME ORA, DOPO STESSA COSA MA TOGLIENDO IL LIVELLO DI MEZZO
# TODO HSP-62 MA CON DROPOUT INIZIALE E WEIGHT DECAY PIù ALTO
PARAMS = {
    "batch_size": 16,
    "lr": 1e-5,
    "max_epochs": 500,
    "weight_decay": 1e-1,
    "patience": 15,
}
neptune_logger.log_hyperparams(params=PARAMS)

NUM_CLASSES = 10
# NUM_CLASSES = 16

# senza stride e più definizione

class GTZANPretrained(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.GTZANDataModule = GTZANDataModule()
        self.GTZANDataModule.setup()
        # self.FMADataLoader = FMADataloader()
        # self.FMADataLoader.setup()
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
        #                               stride=(1, 1), padding=(3, 3), bias=False)
        # self.resnet.avgpool = nn.AdaptiveAvgPool2d((6, 4))
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(2048),
            nn.Linear(self.resnet.fc.in_features, out_features=NUM_CLASSES),
        )
        self.lr = PARAMS["lr"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
        return optimizer

    def forward(self, track):
        predictions = self.resnet(track)
        return predictions

    def on_train_epoch_end(self):
        self.train_dataloader()

    def train_dataloader(self):
        data_loader = self.GTZANDataModule.train_dataloader()
        # data_loader = self.FMADataLoader.train_dataloader()
        return data_loader

    def val_dataloader(self):
        data_loader = self.GTZANDataModule.val_dataloader()
        #
        # data_loader = self.FMADataLoader.val_dataloader()
        return data_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        try:
            self.log("/metrics/batch/train_loss", loss, prog_bar=True, on_epoch=True, on_step=False,
                     sync_dist=True)
        except Exception as e:
            print(loss)
        acc = self.acc(y_pred, y)
        self.log("/metrics/batch/train_acc", acc, prog_bar=True, on_epoch=True, on_step=False,
                 sync_dist=True)
        return {'loss': loss, 'accuracy': acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        try:
            self.log("/metrics/batch/val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        except Exception as e:
            print(loss)
        acc = self.acc(y_pred, y)
        self.log("/metrics/batch/val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return {'val_loss': loss, 'val_accuracy': acc}


checkpoint_callback = ModelCheckpoint(
    monitor='/metrics/batch/val_acc',
    dirpath="/nas/home/ecastelli/thesis/models/Resnet/checkpoint",
    filename='fma-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max',
)

early_stop_callback = EarlyStopping(monitor="/metrics/batch/val_loss",
                                    mode="min",
                                    patience=PARAMS["patience"])

if __name__ == "__main__":
    model = GTZANPretrained()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, logger=neptune_logger,
                         callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=model)
