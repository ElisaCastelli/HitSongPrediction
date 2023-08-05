import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, random_split
import os
from torchmetrics.regression import R2Score
from TracksDataset import TracksDataset
from PreTrainedResNet import PreTrainedResnet
from torchvision.models import ResNet50_Weights, resnet50
import torch.hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"

neptune_logger = NeptuneLogger(
    project="elisacastelli/tesi",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTE3NDY5My02ZjBkLTQxMjktYjY0ZS0wMGI1NmU0NzNjNmYifQ==",
    log_model_checkpoints=True,
)
PARAMS = {
    "batch_size": 64,
    "lr": 1e-4,
    "max_epochs": 150,
    "weight_decay": 1e-2,
    "patience": 40,
}
neptune_logger.log_hyperparams(params=PARAMS)


class MyModel(pl.LightningModule):
    def __init__(self, input_dim=2048, optimizer=None, metrics=None, loss="mae",
                 neuron_parameters={'alpha': 1, 'beta': (1 / 2), 'gamma': (1 / 3)}, layers=5,
                 level_dropout=.25, problem='regression'): #input_dim=2817
        super().__init__()
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.metrics = metrics
        self.neuron_parameters = neuron_parameters
        self.layers = layers
        self.level_dropout = level_dropout
        self.problem = problem
        if self.problem == 'regression':
            self.output_dim = 1
        else:
            self.output_dim = 3
        dataset = TracksDataset()
        self.train_data, self.val_data = random_split(dataset[:4000], [0.8, 0.2]) #TODO crea un dataset bilanciato
        self.pretrainedResnet = PreTrainedResnet.load_from_checkpoint("/nas/home/ecastelli/thesis/models/Resnet/.neptune/Untitled/TES-405/checkpoints/epoch=254-step=12750.ckpt").resnet
        self.pretrainedResnet = nn.Sequential(*list(self.pretrainedResnet.children())[:-1])
        # self.pretrainedResnet.eval()
        # self.pretrainedResnet.freeze()
        # self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # n_hidden_neurons_1 = self.get_hidden_neurons('initial')
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(in_features=self.input_dim, out_features=1024),
            nn.ReLU(),
        )
        # n_hidden_neurons_2 = self.get_hidden_neurons('intermediate')
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
        )
        # n_hidden_neurons_3 = self.get_hidden_neurons('intermediate')
        # self.fc3 = nn.Sequential(
        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.ReLU(),
        # )
        # n_hidden_neurons_4 = self.get_hidden_neurons('final')
        # self.fc4 = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.ReLU(),
        # )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=128),
        )
        self.fc6 = nn.Linear(in_features=128, out_features=self.output_dim)
        if loss == 'mse':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.L1Loss(reduction='mean')
        self.r2score = R2Score()

    def forward(self, images):
        specs = images.squeeze(1)
        audio_emb = self.pretrainedResnet(specs)
        audio_emb = audio_emb.squeeze()
        # year = years.unsqueeze(1)
        # lyrics_emb = torch.tensor(self.sbert_model.encode(lyrics))
        # track = torch.concat((lyrics_emb.cuda(), year.cuda()), dim=1)
        x = self.fc1(audio_emb)
        x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        x = self.fc5(x)
        if self.problem == 'regression':
            x = torch.sigmoid(self.fc6(x))
        else:
            x = torch.softmax(self.fc5(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS["lr"])
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch['popularity']
        y = y.unsqueeze(1)
        # batch['lyrics'], batch['year'],
        y_pred = self(batch['image'])
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/train_loss", loss, prog_bar=True, sync_dist=True)
        acc = self.r2score(y_pred, y)
        self.log("/metrics/batch/train_acc", acc, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'accuracy': acc}

    def validation_step(self, batch, batch_idx):
        y = batch['popularity']
        y = y.unsqueeze(1)
        # batch['lyrics'], batch['year'],
        y_pred = self(batch['image'])
        loss = self.loss(y_pred, y)
        self.log("/metrics/batch/val_loss", loss, prog_bar=True, sync_dist=True)
        acc = self.r2score(y_pred, y)
        self.log("/metrics/batch/val_acc", acc, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'val_accuracy': acc}

    def train_dataloader(self):
        data_loader = DataLoader(dataset=self.train_data, batch_size=PARAMS["batch_size"], num_workers=4)
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(dataset=self.val_data, batch_size=PARAMS["batch_size"], num_workers=4)
        return data_loader


    def get_hidden_neurons(self, n_layer):
        alpha = self.neuron_parameters['alpha']
        beta = self.neuron_parameters['beta']
        gamma = self.neuron_parameters['gamma']

        if n_layer == 'initial':
            n_hidden_neurons = int(round(alpha * self.input_dim))  # 2*input_size
        elif n_layer == 'intermediate':
            n_hidden_neurons = int(round((beta * self.input_dim)))  # beta = 2/3
        else:
            n_hidden_neurons = int(round(gamma * self.input_dim))  # 1/2

        return n_hidden_neurons


if __name__ == "__main__":
    model = MyModel()
    trainer = pl.Trainer(accelerator="gpu", devices=[0, 1], max_epochs=PARAMS["max_epochs"],
                         check_val_every_n_epoch=1, logger=neptune_logger,
                         default_root_dir="/nas/home/ecastelli/thesis/models/checkpoints",
                         callbacks=[EarlyStopping(monitor="/metrics/batch/val_loss",
                                                  mode="min",
                                                  patience=PARAMS["patience"])])
    trainer.fit(model=model)
