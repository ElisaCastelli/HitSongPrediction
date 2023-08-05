import random
import lightning.pytorch as pl
from GTZANDataset import GTZANDataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from random import sample

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/GTZAN/features_30_sec.csv"
lists = {'blues': [],
         'classical': [],
         'country': [],
         'disco': [],
         'hiphop': [],
         'jazz': [],
         'metal': [],
         'pop': [],
         'reggae': [],
         'rock': [],
         }

#TODO FAI 0.6 MA METTI 3 TIPI DI AUGMENTATION


class GTZANDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_list = []
        self.train_data = Dataset()
        self.validation_data = Dataset()
        self.train_aug_data = Dataset()

    def setup(self, stage=None):
        dataframe = pd.read_csv(ANNOTATIONS_FILE)

        for index, row in enumerate(dataframe.itertuples(), 0):
            label = row[-1]
            lists[str(label)].append(row)

        validation_list = []

        for genre in lists.values():
            random.shuffle(genre)
            self.train_list.extend(genre[:60])
            validation_list.extend(genre[60:])

        df_train = pd.DataFrame(self.train_list)
        df_train.drop(columns=['Index'], inplace=True)
        self.train_data = GTZANDataset(df_train)
        df_val = pd.DataFrame(validation_list)
        df_val.drop(columns=['Index'], inplace=True)
        self.validation_data = GTZANDataset(df_val)

    def get_augmented_data(self):
        train_aug_list = []
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_data = pd.DataFrame(train_aug_list)
        train_aug_data.drop(columns=['Index'], inplace=True)
        train_aug_data = GTZANDataset(train_aug_data, augmented=True)
        return train_aug_data

    def train_dataloader(self):
        train_aug_data = self.get_augmented_data()
        train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_data, batch_size=32)
        return val_dataloader


# if __name__ == '__main__':
#     g = GTZANDataModule()

