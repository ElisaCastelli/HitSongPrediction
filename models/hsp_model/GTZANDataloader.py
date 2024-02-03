import random
import lightning.pytorch as pl
from models.hsp_model.GTZANDataset import GTZANDataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/GTZAN/features_30_sec.csv"

''' Lists of songs divided by genres present in GTZANGenre Dataset'''
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


class GTZANDataModule(pl.LightningDataModule):
    """
        Class inheriting from LightningDataModule,
        it has the purpose of standardizing the training, val, test splits, data preparation.
    """
    def __init__(self):
        super().__init__()
        self.train_list = []
        self.validation_list = []
        self.train_data = Dataset()
        self.validation_data = Dataset()
        self.train_aug_data = Dataset()
        self.BATCH_SIZE = 16

    def setup(self, batch_size, stage=None):
        """
            Builds two datasets dividing GTZANGenre songs in a balanced way according to the genre
        """
        self.BATCH_SIZE = batch_size
        # creates a Dataframe with all the songs contained in GTZANGenre
        dataframe = pd.read_csv(ANNOTATIONS_FILE)

        # Divide songs into genres to build a balanced dataset for training and valuation
        for index, row in enumerate(dataframe.itertuples(), 0):
            label = row[-1]
            lists[str(label)].append(row)

        for genre in lists.values():
            random.shuffle(genre)
            self.train_list.extend(genre[:50])
            self.validation_list.extend(genre[50:])

        df_train = pd.DataFrame(self.train_list)
        df_train.drop(columns=['Index'], inplace=True)
        self.train_data = GTZANDataset(df_train)
        df_val = pd.DataFrame(self.validation_list)
        df_val.drop(columns=['Index'], inplace=True)
        self.validation_data = GTZANDataset(df_val)

    def get_augmented_data(self):
        """
            Applies data augmentation creating a dataset in which the songs present in the train list are used four times,
            each time applying a different transformation according to the SpecAugment data augmentation technique.
        """
        train_aug_list = []
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_data = pd.DataFrame(train_aug_list)
        train_aug_data.drop(columns=['Index'], inplace=True)
        train_aug_data = GTZANDataset(train_aug_data, augmented=True)
        return train_aug_data

    def train_dataloader(self):
        """
            The augmented dataset is concatenated with the training dataset that contains the original songs and then is
            loaded with a DataLoader object in batches
        """
        train_aug_data = self.get_augmented_data()
        train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=4, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        """
            The validation dataset is loaded with a DataLoader object in batches
        """
        val_dataloader = DataLoader(self.validation_data, batch_size=self.BATCH_SIZE, num_workers=4)
        return val_dataloader

