import torch
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import Accuracy
from ESC50Dataset import ESC50Dataset
from ESC50DataModule import ESCDataModule
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import tuner
from torchvision.models import ResNet50_Weights
from SEResNet import *

NUM_CLASSES = 50
neptune_logger = NeptuneLogger(
    project="elisacastelli/tesi",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTE3NDY5My02ZjBkLTQxMjktYjY0ZS0wMGI1NmU0NzNjNmYifQ==",
    log_model_checkpoints=False,
)
# TODO PROVA A RIFARLO ANDARE CON DISATTIVARE CROP CHE NEL PAPER NON LO FA E IN DATAAUGMENTATION DEL TRAINING COME IMPOSTATO ORA
PARAMS = {
    "batch_size": 64,
    "lr": 1e-3,
    "max_epochs": 20,
    "weight_decay": 1e-2,
    "patience": 5,
}
# neptune_logger.log_hyperparams(params=PARAMS)


class PreTrainedResnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = ESC50Dataset(None)
        self.data = self.dataset.read_csv_as_dict()
        self.data_module = ESCDataModule(self.data, num_folds=5)
        self.data_module.setup()
        self.validation_step_outputs = []
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
                                      stride=(1, 1), padding=(3, 3), bias=False)
        # TODO STRIDE se ha problemi rimetti a (2, 2)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.resnet.fc.in_features, out_features=NUM_CLASSES),
        )
        self.lr = PARAMS["lr"]

    def configure_optimizers(self):
        print(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"])
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
        return optimizer

    def forward(self, input_data):
        predictions = self.resnet(input_data)
        return predictions

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def train_dataloader(self):
        data_loader = self.data_module.train_dataloader()
        return data_loader

    def val_dataloader(self):
        data_loader = self.data_module.val_dataloader()
        return data_loader

    def manually_cross_entropy(self, net_inputs, y):
        activations = torch.softmax(net_inputs, dim=1)
        y_onehot = F.one_hot(y, num_classes=NUM_CLASSES)
        train_losses = -torch.sum(torch.log(activations) * (y_onehot), dim=1)
        avg_loss = torch.mean(train_losses)
        return avg_loss

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

    def cross_validation(self, num_folds, trainer):
        for fold in range(num_folds):
            trainer.current_fold = fold
            trainer.fit(self)

    def set_current_fold(self, fold):
        self.data_module.current_fold = fold


if __name__ == "__main__":
    model = PreTrainedResnet()
    model.set_current_fold(1)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1,  log_every_n_steps=38,
                         default_root_dir="/nas/home/ecastelli/thesis/models/checkpoints",
                         callbacks=[EarlyStopping(monitor="/metrics/batch/val_loss",
                                                  mode="min",
                                                  patience=PARAMS["patience"])])
    # tun = tuner.Tuner(trainer=trainer)
    # # Run learning rate finder
    # lr_finder = tun.lr_find(model=model)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # # logger = neptune_logger,
    trainer.fit(model=model)

    # for fold in range(1, 6):
    #     model.set_current_fold(fold)
    #     trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=PARAMS["max_epochs"],
    #                          check_val_every_n_epoch=1, logger=neptune_logger,
    #                          default_root_dir="/nas/home/ecastelli/thesis/models/checkpoints",
    #                          callbacks=[EarlyStopping(monitor="/metrics/batch/val_loss", mode="min", patience=30)])
    #     trainer.fit(model=model)
