import random
import lightning.pytorch as pl
from HSPDataset import *
from torch.utils.data import DataLoader
import pandas as pd
import torch

lists = {
    '0': [],
    '1': [],
    '2': [],
    '3': [],
}


def collate_fn(batch):
    return {
        'spectrogram': torch.stack([x['spectrogram'] for x in batch]),
        'year': torch.tensor([x['year'] for x in batch]),
        'lyrics': list([x['lyrics'] for x in batch]),
        'label': torch.tensor([x['label'] for x in batch])
    }


class HSPDataModule(pl.LightningDataModule):
    def __init__(self, problem, language, augmented, annotation_file, num_classes):
        super().__init__()
        self.train_list = []
        self.train_data = None
        self.validation_data = None
        self.train_aug_data = None
        self.problem = problem
        self.num_classes = num_classes
        self.language = language
        self.augmented = augmented
        self.BATCH_SIZE = 64
        self.ANNOTATION_FILE = annotation_file

    def setup(self, batch_size, stage=None):
        self.BATCH_SIZE = batch_size
        dataframe = pd.read_csv(self.ANNOTATION_FILE, index_col=0, encoding='utf8')
        for index, row in enumerate(dataframe.itertuples(), 0):
            if self.language == "en":
                label = row[-5]
            else:
                label = row[-1]
            label = get_class(label, self.num_classes)
            lists[str(label)].append(row)

        validation_list = []
        for pop_class in lists.values():
            random.shuffle(pop_class)
            class_len = len(pop_class)
            q = round(class_len * 75 / 100)
            self.train_list.extend(pop_class[:q])
            validation_list.extend(pop_class[q:])

        random.shuffle(self.train_list)
        df_train = pd.DataFrame(self.train_list)
        df_train.drop(columns=['Index'], inplace=True)
        self.train_data = HSPDataset(subset=df_train, annotation_file=self.ANNOTATION_FILE,
                                     language=self.language, problem=self.problem, num_classes=self.num_classes)
        df_val = pd.DataFrame(validation_list)
        df_val.drop(columns=['Index'], inplace=True)
        self.validation_data = HSPDataset(subset=df_val, annotation_file=self.ANNOTATION_FILE,
                                          language=self.language, problem=self.problem, num_classes=self.num_classes)

    def get_augmented_data(self):
        train_aug_list = []
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_data = pd.DataFrame(train_aug_list)
        train_aug_data.drop(columns=['Index'], inplace=True)
        train_aug_data = HSPDataset(subset=train_aug_data, annotation_file=self.ANNOTATION_FILE, language=self.language,
                                    augmented=True, problem=self.problem, num_classes=self.num_classes)
        return train_aug_data

    def train_dataloader(self):
        train_dataset = self.train_data
        if self.augmented:
            train_aug_data = self.get_augmented_data()
            train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=8, collate_fn=collate_fn,
                                      shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_data, batch_size=self.BATCH_SIZE, num_workers=8,
                                    collate_fn=collate_fn)
        return val_dataloader

    def get_shortest(self):
        dataframe = HSPDataset(subset=pd.read_csv(self.ANNOTATION_FILE), augmented=False, problem='c')
        length = len(dataframe)
        min = 700000
        for i in range(0, length - 1):
            audio = dataframe[i]
            if int(len(audio)) < min:
                min = int(len(audio))

        print(min)
