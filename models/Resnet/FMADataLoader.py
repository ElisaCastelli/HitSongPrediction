import random
import lightning.pytorch as pl
from FMADataset import FMADataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/FMA/tracks.csv"


class FMADataloader(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataframe = pd.DataFrame()
        self.train_data = Dataset()
        self.val_data = Dataset()
        self.test_data = Dataset()
        self.train_aug_data = Dataset()

    def setup(self, stage=None):
        dataframe = pd.read_csv(ANNOTATIONS_FILE)
        medium_dataset = dataframe[dataframe['set.1'] == "medium"]
        small_dataset = dataframe[dataframe['set.1'] == "small"]
        medium_dataset = pd.concat([medium_dataset, small_dataset])

        dataframe_training = medium_dataset[medium_dataset['set'] == "training"]
        dataframe_validation = medium_dataset[medium_dataset['set'] == "validation"]
        dataframe_test = medium_dataset[medium_dataset['set'] == "test"]
        self.train_dataframe = dataframe_training

        self.train_data = FMADataset(subset=dataframe_training)
        self.val_data = FMADataset(subset=dataframe_validation)
        self.test_data = FMADataset(subset=dataframe_test)

    def get_augmented_data(self):
        train_aug_data = pd.concat([self.train_dataframe, self.train_dataframe, self.train_dataframe])
        # train_aug_data = pd.DataFrame(train_aug_list)
        # train_aug_data.drop(columns=['Index'], inplace=True)
        train_aug_data = FMADataset(train_aug_data, augmented=True)
        return train_aug_data

    def train_dataloader(self):
        train_aug_data = self.get_augmented_data()
        train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_data, batch_size=16, num_workers=4)
        return val_dataloader
