import lightning.pytorch as pl
from sklearn.model_selection import KFold
from ESC50Dataset import ESC50Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch


class ESCDataModule(pl.LightningDataModule):
    def __init__(self, data, num_folds):
        super().__init__()
        self.fold_data = data
        self.num_folds = num_folds
        self.current_fold = 0

    def setup(self, stage=None):
        kfold = KFold(n_splits=self.num_folds, shuffle=False)
        for fold, (train_indices, val_indices) in enumerate(kfold.split(self.fold_data)):
            train_data = pd.DataFrame()
            val_data = pd.DataFrame()
            for i in train_indices:
                train_data = pd.concat([train_data, pd.DataFrame.from_dict(self.fold_data[i])], ignore_index=True)
            for i in val_indices:
                val_data = pd.concat([val_data, pd.DataFrame.from_dict(self.fold_data[i])], ignore_index=True)

            train_dataset = ESC50Dataset(train_data)
            train_augmented = ESC50Dataset(train_data[:800], augmented=True)
            val_dataset = ESC50Dataset(val_data)

            setattr(self, f'train_dataset_fold{fold+1}', train_dataset)
            setattr(self, f'train_aug_dataset_fold{fold + 1}', train_augmented)
            setattr(self, f'val_dataset_fold{fold+1}', val_dataset)

    # GET train dataloader given an index (folder)
    def train_dataloader(self):
        train_dataset = getattr(self, f'train_dataset_fold{self.current_fold}')
        train_aug_dataset = getattr(self, f'train_aug_dataset_fold{self.current_fold}')
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_aug_dataset])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataloader

    # GET val dataloader given an index
    def val_dataloader(self):
        val_dataset = getattr(self, f'val_dataset_fold{self.current_fold}')
        val_dataloader = DataLoader(val_dataset, batch_size=32)
        return val_dataloader
