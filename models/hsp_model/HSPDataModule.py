import random
import lightning.pytorch as plight
from models.hsp_model.HSPDataset import *
from torch.utils.data import DataLoader
import polars as pl
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
        'lyrics': [x['lyrics'] for x in batch], #list()
        'label': torch.tensor([x['label'] for x in batch])
    }


class HSPDataModule(plight.LightningDataModule):
    def __init__(self, problem, language, annotation_file, num_classes, augmented=True,):
        super().__init__()
        self.train_list = []
        self.train_data = None
        self.validation_data = None
        self.train_aug_data = None
        self.problem = problem
        self.num_classes = num_classes
        self.language = language
        self.augmented = augmented
        self.BATCH_SIZE = 32
        self.ANNOTATION_FILE = annotation_file

    def setup(self, batch_size, stage=None):
        self.BATCH_SIZE = batch_size
        dataframe = pl.read_parquet(self.ANNOTATION_FILE)
        #dataframe = pl.read_csv(self.ANNOTATION_FILE, index_col=0, encoding='utf8')

        for row in dataframe.rows(named=True):
            label = row['popularity']
            label = get_class(label, self.num_classes)
            lists[str(label)].append(row)


        # stratification of popularity 
            
        validation_list = []
        for pop_class in lists.values():
            random.shuffle(pop_class)
            class_len = len(pop_class)
            q = round(class_len * 75 / 100)
            self.train_list.extend(pop_class[:q])
            validation_list.extend(pop_class[q:])

        random.shuffle(self.train_list)
        df_train = pl.DataFrame(self.train_list)
        df_train=df_train.drop("Index")
        self.train_data = HSPDataset(subset=df_train, annotation_file=self.ANNOTATION_FILE,
                                     language=self.language, problem=self.problem, num_classes=self.num_classes)
        df_val = pl.DataFrame(validation_list)
        df_val=df_val.drop("Index")
        self.validation_data = HSPDataset(subset=df_val, annotation_file=self.ANNOTATION_FILE,
                                          language=self.language, problem=self.problem, num_classes=self.num_classes)

    def get_augmented_data(self):
        train_aug_list = []
        train_aug_list.extend(self.train_list)
        train_aug_list.extend(self.train_list)
        train_aug_data = pl.DataFrame(train_aug_list)
        train_aug_data=train_aug_data.drop("Index")
        train_aug_data = HSPDataset(subset=train_aug_data, annotation_file=self.ANNOTATION_FILE, language=self.language,
                                    augmented=True, problem=self.problem, num_classes=self.num_classes)
        return train_aug_data

    def train_dataloader(self):
        train_dataset = self.train_data
        if self.augmented:
            train_aug_data = self.get_augmented_data()
            train_dataset = torch.utils.data.ConcatDataset([self.train_data, train_aug_data])
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=8, collate_fn=collate_fn,
                                      shuffle=True,pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_data, batch_size=self.BATCH_SIZE, num_workers=8,
                                    collate_fn=collate_fn,pin_memory=True)
        return val_dataloader
