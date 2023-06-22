import os
import re
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from torchvision import transforms
# from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class MyModel(pl.LightningModule):
    def __init__(self, input_dim=None,
                 output_dim=1, optimizer=None, metrics=None, loss="mse",
                 scale_pred=False, scale_target=True, add_early_stopping=False,
                 audio_model=False, weights_path=None,
                 saved_weights=True, load_weights=False,
                 neuron_parameters={'alpha': 1, 'beta': (1/2), 'gamma': (1/3)}, layers=5, initialization='he_normal',
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
        self.bert = None

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

    def training_step(self):
        pass

    def test_step(self):
        pass

    def validation_step(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_hidden_neurons(self, n_layer):

        alpha = self.neuron_parameters['alpha']  # 5
        beta = self.neuron_parameters['beta']  # 6/3
        gamma = self.neuron_parameters['gamma']  # 3

        if n_layer == 'initial':
            n_hidden_neurons = int(round(alpha * self.input_size))  # 2*input_size
        elif n_layer == 'intermediate':
            n_hidden_neurons = int(round((beta * self.input_size)))  # beta = 2/3
        else:
            n_hidden_neurons = int(round(gamma * self.input_size))  # 1/2

        return n_hidden_neurons
