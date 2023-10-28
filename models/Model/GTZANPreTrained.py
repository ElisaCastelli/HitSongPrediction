import torch.nn as nn
import torch
import torchvision.models as models
from torchmetrics import Accuracy
from GTZANDataloader import GTZANDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.models import ResNet50_Weights

''' NeptuneLogger object to log metrics and parameters'''
neptune_logger = NeptuneLogger(
    project="elishcastle/HSP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTkxYWQyNi1kMWM3LTRiNzYtYWJhNS05ZThmNWZiNmMwOTIifQ==",
)

''' Model Parameters'''
PARAMS = {
    "batch_size": 16,
    "lr": 1e-5,
    "max_epochs": 200,
    "weight_decay": 1e-1,
    "patience": 15,
}
''' Add to NeptuneLogger the hyperparameters'''
neptune_logger.log_hyperparams(params=PARAMS)

''' Number of class (music genre) the model classifies '''
NUM_CLASSES = 10

''' Definition of the ModelCheckpoint used to save the best model weights according to the accuracy reached '''
checkpoint_callback = ModelCheckpoint(
    monitor='/metrics/batch/val_acc',
    dirpath="/nas/home/ecastelli/thesis/models/Model/checkpoint",
    filename='gtzan-{epoch:02d}-{val_acc:.2f}',
    save_top_k=2,
    mode='max',
)
''' Definition of the EarlyStopping used to stop the execution when the validation loss value does not improve '''
early_stop_callback = EarlyStopping(monitor="/metrics/batch/val_loss",
                                    mode="min",
                                    patience=PARAMS["patience"])


class GTZANPretrained(pl.LightningModule):
    """ Class inheriting from LightningModule, it has the purpose of creating a model
     pre-trained using GTZANGenre dataset that will be used to extract music audio embeddings """
    def __init__(self):
        """ Init method of the class GTZANPreTrained """
        super().__init__()
        self.GTZANDataModule = GTZANDataModule()
        self.GTZANDataModule.setup(PARAMS["batch_size"])
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
        """
            Initializes the optimizer used during the training process
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
        return optimizer

    def forward(self, track):
        """
            Applies the model

            Input: the batch of tracks to be analyzed

            Output: the model prediction
        """
        predictions = self.resnet(track)
        return predictions

    def on_train_epoch_end(self):
        """
            Changes the data augmentation applied to the training dataset portion at each training epoch end
        """
        self.train_dataloader()

    def train_dataloader(self):
        """
            Returns the training data loader
        """
        data_loader = self.GTZANDataModule.train_dataloader()
        return data_loader

    def val_dataloader(self):
        """
            Returns the validation data loader
        """
        data_loader = self.GTZANDataModule.val_dataloader()
        return data_loader

    def training_step(self, batch, batch_idx):
        """
            Starting from each batch of audio it takes mel spectrograms and targets. It applies the ResNet model to the audio,
            gets the result and compute the loss and the accuracy.
        """
        x, y = batch
        y_pred = self(x)  # calls forward
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
        """
            Starting from each batch of audio it takes mel spectrograms and targets. It applies the ResNet model to the audio,
            gets the result and compute the loss and the accuracy.
        """
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


if __name__ == "__main__":
    model = GTZANPretrained()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, logger=neptune_logger,
                         callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=model)
