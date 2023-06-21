import os
import torch
from torch import nn
import torch.nn.functional as f
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = f.mse_loss(x_hat, x)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def print_hyper_param(self, checkpoint):
        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print(checkpoint["hyper_parameters"])


def init_with_other_params():
    # if you train and save the model like this it will use these values when loading
    # the weights. But you can overwrite this
    # LitModel(in_dim=32, out_dim=10)
    # uses in_dim=32, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH)
    # uses in_dim=128, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)
    pass


# Disable checkpointing
# trainer = Trainer(enable_checkpointing=False)

# Resume training state
# model = LitModel()
# trainer = Trainer()
#
# automatically restores model, epoch, step, LR schedulers, etc...
# trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

transform = transforms.ToTensor()  # {"learning_rate": the_value, "another_parameter": the_other_value}
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)

model = LitAutoEncoder(Encoder(), Decoder())
# train with both splits
trainer = Trainer(
    default_root_dir="/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/saved_models")  # salva il checkpoint
# trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")]) imposta earlystopping
# OPPURE
# early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
# trainer = Trainer(callbacks=[early_stop_callback])
trainer.fit(model, train_loader, valid_loader)


# per ricominciare dal checkpoint
# model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
# model.eval()

# predict with the model
# y_hat = model(x)

class BertMNLIFinetuner(LightningModule):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)

    model = ImagenetTransferLearning.load_from_checkpoint(PATH)
    model.freeze()

    x = some_images_from_cifar10()
    predictions = model(x)