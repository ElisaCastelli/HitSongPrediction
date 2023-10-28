import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split
from TracksDataset import TracksDataset
from torchmetrics.regression import R2Score
from torchmetrics import Accuracy, Recall, F1Score, Precision
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
from GTZANPreTrained import GTZANPretrained
from torchvision import transforms
import torch.nn.functional as F
from HSPDataModule import HSPDataModule
from sentence_transformers import SentenceTransformer
from models_configurations import *

neptune_logger = NeptuneLogger(
    project="elishcastle/HSP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTkxYWQyNi1kMWM3LTRiNzYtYWJhNS05ZThmNWZiNmMwOTIifQ==",
)

PARAMS = {
    "batch_size": 64,
    "lr": 1e-5,
    "max_epochs": 300,
    "weight_decay": 1e-2,
    "patience": 30,
}
neptune_logger.log_hyperparams(params=PARAMS)

NUM_CLASSES = 4


class HSPModel(pl.LightningModule):
    def __init__(self, language, problem, augmented):
        """
            Builder
        """
        super().__init__()
        self.language = language
        if self.language == "en":
            self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 768 --> 2817 total
            self.annotation_file = "/nas/home/ecastelli/thesis/Billboard/CSV/SPD_en_no_dup.csv"
        else:
            self.sbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
            # self.sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            self.annotation_file = "/nas/home/ecastelli/thesis/Billboard/CSV/SPD_all_lang_not_updated.csv"
        self.tensor_transform = transforms.ToTensor()
        self.problem = problem
        if problem == 'r':
            self.loss = nn.L1Loss()
            self.loss2 = nn.MSELoss()
            self.acc = R2Score()
        else:
            self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
            self.loss = nn.CrossEntropyLoss(reduction="mean")
            self.recall = MulticlassRecall(num_classes=NUM_CLASSES)
            self.f1score = MulticlassF1Score(num_classes=NUM_CLASSES)
            self.precision = MulticlassPrecision(num_classes=NUM_CLASSES)

        self.datamodule = HSPDataModule(problem=problem, augmented=augmented)
        self.datamodule.setup(PARAMS["batch_size"])

        # checkpoint_path = "/nas/home/ecastelli/thesis/models/Model/checkpoint/GTZAN_HPSS-epoch=50-/metrics/batch/val_acc=0.77.ckpt"
        checkpoint_path = "/nas/home/ecastelli/thesis/models/Model/checkpoint/NuovoGTZAN-epoch=24-val_acc=0.00.ckpt"
        self.resnet = GTZANPretrained.load_from_checkpoint(checkpoint_path).resnet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
        #                               stride=(2, 2), padding=(3, 3), bias=False)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-4])
        # avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.resnet.add_module("avgpool", avg)  # -> 1024 embeddings
        model_name = self.problem + "-" + self.language
        self.layers = select_model(model_name)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
        # optimizer = torch.optim.SGD(self.parameters(), lr=PARAMS["lr"], weight_decay=1e-2)
        return optimizer

    def forward(self, spectrogram, lyrics, year):  # lyrics,
        lyrics_emb = self.sbert_model.encode(lyrics)
        lyrics_emb = self.tensor_transform(lyrics_emb)
        lyrics_emb = lyrics_emb.squeeze()
        # lyrics_emb = F.normalize(lyrics_emb, p=2, dim=1)
        self.resnet.eval()
        with torch.no_grad():
            embeddings = self.resnet(spectrogram)
        embeddings = embeddings.squeeze()
        year = year.unsqueeze(1)
        track = torch.concat((embeddings, lyrics_emb.cuda(), lyrics_emb.cuda(), year), dim=1)  # lyrics_emb.cuda(),
        # track = torch.concat((embeddings, year), dim=1)
        x = self.layers(track)
        return x

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        spec = batch["spectrogram"]
        year = batch["year"]
        lyrics = batch["lyrics"]
        if self.problem == 'r':
            y = torch.div(y, 100)
            y = y.unsqueeze(1)
        y_pred = self(spec, lyrics, year)  # lyrics,
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=PARAMS["batch_size"])
        acc = self.acc(y_pred, y)
        self.log("/metrics/batch/train_acc", acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=PARAMS["batch_size"])
        if self.problem == 'c':
            recall = self.recall(y_pred, y)
            self.log("/metrics/batch/train_recall", recall, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
            precision = self.precision(y_pred, y)
            self.log("/metrics/batch/train_precision", precision, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
            f1score = self.f1score(y_pred, y)
            self.log("/metrics/batch/train_f1score", f1score, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
        else:
            loss2 = self.loss2(y_pred, y)
            self.log("/metrics/batch/train_loss2", loss2, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=PARAMS["batch_size"])
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        spec = batch["spectrogram"]
        year = batch["year"]
        lyrics = batch["lyrics"]
        if self.problem == 'r':
            y = torch.div(y, 100)
            y = y.unsqueeze(1)
        y_pred = self(spec, lyrics, year)  # lyrics,
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=PARAMS["batch_size"])
        acc = self.acc(y_pred, y)
        self.log("/metrics/batch/val_acc", acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=PARAMS["batch_size"])
        if self.problem == 'c':
            recall = self.recall(y_pred, y)
            self.log("/metrics/batch/val_recall", recall, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
            precision = self.precision(y_pred, y)
            self.log("/metrics/batch/val_precision", precision, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
            f1score = self.f1score(y_pred, y)
            self.log("/metrics/batch/val_f1score", f1score, prog_bar=True, on_epoch=True, on_step=False,
                     batch_size=PARAMS["batch_size"])
        else:
            loss2 = self.loss2(y_pred, y)
            self.log("/metrics/batch/val_loss2", loss2, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=PARAMS["batch_size"])
        return {'val_loss': loss}

    def train_dataloader(self):
        if self.dataset == 'BB':
            data_loader = self.datamodule.train_dataloader()
        else:
            data_loader = DataLoader(dataset=self.train_data, batch_size=PARAMS["batch_size"])
        return data_loader

    def val_dataloader(self):
        if self.dataset == 'BB':
            data_loader = self.datamodule.val_dataloader()
        else:
            data_loader = DataLoader(dataset=self.val_data, batch_size=PARAMS["batch_size"])
        return data_loader

    def freeze_pretrained(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        result = sum_embeddings / sum_mask
        return result.numpy().tolist()

    def gen_embedding(self, text, model, tokenizer):
        # Tokenize the texts
        encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Encode the tokenized data with model
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Pool the outputs into a single vector
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings


# checkpoint_callback = ModelCheckpoint(
#     monitor='/metrics/batch/val_acc',
#     dirpath="/nas/home/ecastelli/thesis/models/Model/checkpoint/TransferLearning",
#     filename='FIRST-TRY-{epoch:02d}-{/metrics/batch/val_acc:.2f}',
#     save_top_k=3,
#     mode='max',
# )

early_stop_callback = EarlyStopping(monitor="/metrics/batch/val_loss",
                                    mode="min",
                                    patience=PARAMS["patience"])

if __name__ == "__main__":
    '''
        Problem:
            - Classification --> 'c'
            - Regression --> 'r'
        Language:
            - English --> 'en'
            - Multilingual --> 'mul'
    '''
    model = HSPModel(problem='c', language="en", augmented=True)
    model.freeze_pretrained()
    trainer = pl.Trainer(accelerator="gpu", devices=[0, 1], max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, logger=neptune_logger,
                         callbacks=[early_stop_callback], strategy='ddp_find_unused_parameters_true')
    trainer.fit(model=model)
