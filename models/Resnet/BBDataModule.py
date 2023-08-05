import random
import lightning.pytorch as pl
from BBDataset import BBDataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_FINAL.csv"
lists = {'0': [],
         '1': [],
         '2': [],
         '3': [],
         }

class BBDataModule(pl.LightningDataModule):
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
        for pop_class in lists.values():
            random.shuffle(pop_class)
            class_len = len(pop_class)
            q = round(class_len * 75 / 100)
            self.train_list.extend(pop_class[:q])
            validation_list.extend(pop_class[q:])

        df_train = pd.DataFrame(self.train_list)
        df_train.drop(columns=['Index'], inplace=True)
        self.train_data = BBDataset(subset=df_train)
        df_val = pd.DataFrame(validation_list)
        df_val.drop(columns=['Index'], inplace=True)
        self.validation_data = BBDataset(subset=df_val)

    def get_augmented_data(self):
        train_aug_list = []
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_data = pd.DataFrame(train_aug_list)
        train_aug_data.drop(columns=['Index'], inplace=True)
        train_aug_data = BBDataset(train_aug_data, augmented=True)
        return train_aug_data

    def train_dataloader(self):
        # train_aug_data = self.get_augmented_data()
        # train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_data, batch_size=32)
        return val_dataloader
