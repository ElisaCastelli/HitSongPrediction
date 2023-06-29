import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import lightning.pytorch as pl
from sentence_transformers import SentenceTransformer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # nepture

from dataset_mgmt import *
from analysis.audio_processing import AudioPreProcessing
from track import Track


class MyModel(pl.LightningModule):
    def __init__(self, input_dim=None,
                 output_dim=1, optimizer=None, metrics=None, loss="mse",
                 scale_pred=False, scale_target=True,
                 audio_model=False, add_earlyStopping=False, weights_path=None,
                 saved_weights=True, load_weights=False,
                 neuron_parameters={'alpha': 1, 'beta': (1 / 2), 'gamma': (1 / 3)}, layers=5,
                 initialization='he_normal',
                 level_dropout=.25, problem='regression'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.scale_pred = scale_pred
        self.scale_target = scale_target
        self.add_earlyStopping = add_earlyStopping
        self.audio_model = audio_model
        self.neuron_parameters = neuron_parameters
        self.layers = layers
        self.initialization = initialization
        self.level_dropout = level_dropout
        self.problem = problem

        # input dim è da calcolare sulla base degli embeddings
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove last classification layer
        # TODO remove stride of 2 in the first conv layer

        # Load the model
        self.sbert_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')

        n_hidden_neurons_1 = self.get_hidden_neurons('initial')
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=n_hidden_neurons_1),
            nn.ReLU(),
            nn.Dropout(p=self.level_dropout)
        )
        n_hidden_neurons_2 = self.get_hidden_neurons('intermediate')
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=n_hidden_neurons_2),
            nn.ReLU(),
            nn.Dropout(p=self.level_dropout)
        )
        n_hidden_neurons_3 = self.get_hidden_neurons('intermediate')
        self.dense3 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=n_hidden_neurons_3),
            nn.ReLU(),
            nn.Dropout(p=self.level_dropout)
        )
        n_hidden_neurons_4 = self.get_hidden_neurons('final')
        self.dense4 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=n_hidden_neurons_4),
            nn.ReLU(),
            nn.Dropout(p=self.level_dropout)
        )

        if self.scale_target and self.problem == 'regression':
            activate_funct = nn.Sigmoid()
        elif not self.scale_target and self.problem == 'regression':
            # era Linear in tensorflow
            activate_funct = None
        elif self.problem == 'classification' and self.scale_target:  # Classification
            activate_funct = nn.Softmax()
        else:
            activate_funct = None

        self.dense5 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.output_dim),
            activate_funct
        )

    def forward(self, track):
        # Encode query
        query_emb = self.sbert_model.encode(track.lyrics)  # 768 vector dimension
        print(query_emb.shape)
        print(track.mel_spectrogram.shape)
        spec_features = self.resnet(track.mel_spectrogram)
        spec_features = torch.flatten(spec_features, start_dim=1)
        print(spec_features.shape)
        features = torch.concat((query_emb, spec_features, track.year), dim=0)
        x = self.dense1(features)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

    def training_step(self, batch):
        x, y = batch    # x è il vettore di oggetti e y è il vettore di target
        y_pred = self(x)
        # imposta la loss e fai return
        pass

    def test_step(self):
        pass

    def validation_step(self):
        # vedi che metriche devi aggiungere
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_hidden_neurons(self, n_layer):

        alpha = self.neuron_parameters['alpha']  # 5
        beta = self.neuron_parameters['beta']  # 6/3
        gamma = self.neuron_parameters['gamma']  # 3

        if n_layer == 'initial':
            n_hidden_neurons = int(round(alpha * self.input_dim))  # 2*input_size
        elif n_layer == 'intermediate':
            n_hidden_neurons = int(round((beta * self.input_dim)))  # beta = 2/3
        else:
            n_hidden_neurons = int(round(gamma * self.input_dim))  # 1/2

        return n_hidden_neurons


if __name__ == "__main__":
    # "/nas/home/ecastelli/thesis/Audio/"
    m = DatasetMGMT()

    a = AudioPreProcessing("5qljLQuKnNJf4F4vfxQB0V")
    a.resample()
    mel = a.get_log_mel_spectrogram(print_image=False)
    lyrics = m.get_info_by_id(id='5qljLQuKnNJf4F4vfxQB0V')['lyrics'][0]
    # model = MyModel(input_dim=224)
    # model.training_step(lyrics, input_tensor, 2010)
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last classification layer
    emb_features = resnet(a.three_channels_mel())
    print(emb_features)
