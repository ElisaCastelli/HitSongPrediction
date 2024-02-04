import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import lightning.pytorch as plight
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchmetrics.regression import R2Score
from torchmetrics import Accuracy, Recall, F1Score, Precision
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
from models.hsp_model.GTZANPreTrained import GTZANPretrained
from torchvision import transforms
from models.hsp_model.HSPDataModule import HSPDataModule
#from sentence_transformers import SentenceTransformer
from models.hsp_model.models_configurations import *

''' NeptuneLogger object to log metrics and parameters'''
""" neptune_logger = NeptuneLogger(
    project="elishcastle/HSP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTkxYWQyNi1kMWM3LTRiNzYtYWJhNS05ZThmNWZiNmMwOTIifQ==",
) """

''' Model Parameters'''
PARAMS = {
    "batch_size": 64,
    "lr": 1e-5,
    "max_epochs": 300,
    "weight_decay": 1e-2,
    "patience": 30,
}
#neptune_logger.log_hyperparams(params=PARAMS)

''' Definition of the ModelCheckpoint used to save the best model weights according to the accuracy reached '''
# checkpoint_callback = ModelCheckpoint(
#     monitor='/metrics/batch/val_acc',
#     dirpath="/nas/home/ecastelli/thesis/models/Model/checkpoint/TransferLearning",
#     filename='FIRST-TRY-{epoch:02d}-{/metrics/batch/val_acc:.2f}',
#     save_top_k=3,
#     mode='max',
# )

''' Definition of the EarlyStopping used to stop the execution when the validation loss value does not improve '''
early_stop_callback = EarlyStopping(monitor="/metrics/batch/val_loss",
                                    mode="min",
                                    patience=PARAMS["patience"])

annotations_local={
    "en":"Datasets/SPD_english.parquet",
    "mul":"Datasets/SPD_multilingual.parquet"
}

annotations_nas={
    "en":"/nas/home/ecastelli/thesis/Datasets/SPD_en_no_dup.csv",
    "mul":"/nas/home/ecastelli/thesis/Datasets/SPD_all_lang_no_dup.csv"
}

sentence_bert={
    "en":"sentence-transformers/all-mpnet-base-v2",
    "mul":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    #"mul":"sentence-transformers/multi-qa-mpnet-base-dot-v1"
}

class HSPModel(plight.LightningModule):
    """
        Class inheriting from LightningModule, it has the purpose of creating a model
        that will be used to predict songs popularity
    """

    def __init__(self, language, problem, num_classes=4):
        """
            Builder to set all the model parameter according to language selected and problem to solve
        """
        super().__init__()
        self.annotation_file=annotations_local[language]
        self.sbert_model=sentence_bert[language]
        self.tensor_transform = transforms.ToTensor()
        self.language = language
        if problem == 'r':  # Regression
            self.loss = nn.L1Loss()
            self.loss2 = nn.MSELoss()
            self.acc = R2Score()
            self.num_classes = 4 # used to do stratification on popularity
        else:  # Classification
            self.num_classes = num_classes
            self.acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.loss = nn.CrossEntropyLoss(reduction="mean")
            self.recall = MulticlassRecall(num_classes=num_classes)
            self.f1score = MulticlassF1Score(num_classes=num_classes)
            self.precision = MulticlassPrecision(num_classes=num_classes)
        self.problem = problem
        self.datamodule = HSPDataModule(problem=problem, language=language, num_classes=self.num_classes, annotation_file=self.annotation_file)
        self.datamodule.setup(PARAMS["batch_size"])

        # LOAD PRE-TRAINED RESNET-50 FINE-TUNED ON GTZAN GENRE

        # checkpoint_path =
        # "/nas/home/ecastelli/thesis/models/Model/checkpoint/GTZAN_HPSS-epoch=50-/metrics/batch/val_acc=0.77.ckpt"
        checkpoint_path = "/nas/home/ecastelli/thesis/models/Model/checkpoint/NuovoGTZAN-epoch=24-val_acc=0.00.ckpt"
        self.resnet = GTZANPretrained.load_from_checkpoint(checkpoint_path).resnet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
        #                               stride=(2, 2), padding=(3, 3), bias=False)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-4])
        # avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.resnet.add_module("avgpool", avg)  # -> 1024 embeddings

        # Load the model configuration according to the problem and the language used
        model_name = self.problem + "-" + self.language
        self.layers = select_model(model_name, num_classes)
        if self.layers is None:
            print("Error selecting model configuration!")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
        return optimizer

    def forward(self, spectrogram, lyrics, year):
        """
            Applies the model

            Input: the batch of tracks, lyrics and release year to be analyzed

            Output: the model prediction
        """
        lyrics_emb = self.sbert_model.encode(lyrics)
        lyrics_emb = self.tensor_transform(lyrics_emb)
        lyrics_emb = lyrics_emb.squeeze()
        if self.language == "mul":
            lyrics_emb = F.normalize(lyrics_emb, p=2, dim=1)  # Sentence-BERT multilingual does not normalize embeddings
        self.resnet.eval()
        with torch.no_grad():
            embeddings = self.resnet(spectrogram)
        embeddings = embeddings.squeeze()
        year = year.unsqueeze(1)
        # embeddings concatenation changes if we are evaluating the impact of only audio embeddings,
        # single text embeddings and weighting text embeddings taking them twice (default)
        if self.problem == "c-onlyaudio-en":
            track = torch.concat((embeddings, year), dim=1)
        elif self.problem == "c-onetext-en":
            track = torch.concat((embeddings, lyrics_emb.cuda(), year), dim=1)
        else:
            track = torch.concat((embeddings, lyrics_emb.cuda(), lyrics_emb.cuda(), year), dim=1)
        x = self.layers(track)
        return x

    def training_step(self, batch, batch_idx):
        """
            Starting from each batch of audio it takes audio, lyrics, release years and targets.
            It applies the model, gets the result and compute the loss and the accuracy.
        """
        y = batch["label"]
        spec = batch["spectrogram"]
        year = batch["year"]
        lyrics = batch["lyrics"]
        if self.problem == 'r':
            y = torch.div(y, 100)
            y = y.unsqueeze(1)
        y_pred = self(spec, lyrics, year)
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
        """
            Starting from each batch of audio it takes audio, lyrics, release years and targets.
            It applies the model, gets the result and compute the loss and the accuracy.
        """
        y = batch["label"]
        spec = batch["spectrogram"]
        year = batch["year"]
        lyrics = batch["lyrics"]
        if self.problem == 'r':
            y = torch.div(y, 100)
            y = y.unsqueeze(1)
        y_pred = self(spec, lyrics, year)
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
        """
            Returns the training data loader
        """
        data_loader = self.datamodule.train_dataloader()
        return data_loader

    def val_dataloader(self):
        """
            Returns the validation data loader
        """
        data_loader = self.datamodule.val_dataloader()
        return data_loader

    def freeze_pretrained(self):
        for param in self.resnet.parameters():
            param.requires_grad = False


def hit_song_prediction(problem, language, num_classes):
    """
        Problem:
            - Classification --> 'c'
            - Regression --> 'r'
        Language:
            - English --> 'en'
            - Multilingual --> 'mul'
        Trainer devices: if multi GPU devices [0, 1] and add strategy='ddp_find_unused_parameters_true'
    """
    if problem not in ["c","r"] or language not in ["en","mul"]:
        print("Check the parameters!\nproblem: you can choose between \"c\" or \"r\" for classification or regression\
              \nlanguage: you can choose between \"en\" or \"mul\" for english or multilingual")
        exit()
    model = HSPModel(problem=problem, language=language, num_classes=num_classes)
    model.freeze_pretrained()

    # When using multi GPU change the parameter devices=[0,1] and add strategy='ddp_find_unused_parameters_true'

    trainer = plight.Trainer(accelerator="gpu", devices=[0, 1], max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, 
                         callbacks=[early_stop_callback], strategy='ddp_find_unused_parameters_true')
    #logger=neptune_logger,
    trainer.fit(model=model)
