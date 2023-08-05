import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split
from TracksDataset import TracksDataset
from GTZANPreTrained import GTZANPretrained

neptune_logger = NeptuneLogger(
    project="elishcastle/HSP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTkxYWQyNi1kMWM3LTRiNzYtYWJhNS05ZThmNWZiNmMwOTIifQ==",
)

PARAMS = {
    "batch_size": 128,
    "lr": 1e-3,
    "max_epochs": 100,
    "weight_decay": 1e-2,
    "patience": 30,
}
neptune_logger.log_hyperparams(params=PARAMS)


class FineTuningResNet(pl.LightningModule):
    def __init__(self):
        """
            Builder
        """
        super().__init__()
        dataset = TracksDataset()
        self.train_data, self.val_data = random_split(dataset, [0.75, 0.25])
        # self.resnet = resnet50(weights=None)
        checkpoint_path = "/nas/home/ecastelli/thesis/models/Resnet/.neptune/Untitled/HSP-24/checkpoints/epoch=97-step=4312.ckpt"
        self.resnet = GTZANPretrained.load_from_checkpoint(checkpoint_path).resnet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss = nn.L1Loss()

    def configure_optimizers(self):
        """
            Set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer

    def forward(self, spectrogram, year):
        """
          Forward pass
        """
        # self.resnet.eval()
        embeddings = self.resnet(spectrogram)
        embeddings = embeddings.squeeze()
        # year = year.unsqueeze(1)
        # track = torch.concat((embeddings, year), dim=1)
        x = self.layers(embeddings)
        return x

    def training_step(self, batch, batch_idx):
        y = batch['popularity']
        y = y.unsqueeze(1)
        y_pred = self(batch['image'], batch['year'])
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # acc = self.r2score(y_pred, y)
        # self.log("/metrics/batch/train_acc", acc, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch['popularity']
        y = y.unsqueeze(1)
        y_pred = self(batch['image'], batch['year'])
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # acc = self.r2score(y_pred, y)
        # self.log("/metrics/batch/val_acc", acc, prog_bar=True, sync_dist=True)
        return {'val_loss': loss}

    def train_dataloader(self):
        data_loader = DataLoader(dataset=self.train_data, batch_size=PARAMS["batch_size"])
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(dataset=self.val_data, batch_size=PARAMS["batch_size"])
        return data_loader

    def freeze_pretrained(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

checkpoint_callback = ModelCheckpoint(
    monitor='/metrics/batch/val_loss',
    dirpath="/nas/home/ecastelli/thesis/models/Resnet/checkpoint/TransferLearning",
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

early_stop_callback = EarlyStopping(monitor="/metrics/batch/val_loss",
                                    mode="min",
                                    patience=PARAMS["patience"])

if __name__ == "__main__":
    model = FineTuningResNet()
    model.freeze_pretrained()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, logger=neptune_logger,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model)
    # torch.save(model.resnet, "my_resnet.pt")
