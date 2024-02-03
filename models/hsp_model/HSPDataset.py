import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from models.hsp_model.audio_processing import *

SAMPLE_RATE = 22050
# SAMPLE_RATE = 44100


AUDIO_DIR = "/nas/home/ecastelli/thesis/Audio/"


def get_class(popularity, num_division):
    """
        Based on the number of class in which we want to classify the songs popularity,
        taking the popularity value of a song this method returns the associated popularity class
    """
    if num_division == 4:
        if popularity < 25:
            pop_class = 0
        elif 25 <= popularity < 50:
            pop_class = 1
        elif 50 <= popularity < 75:
            pop_class = 2
        else:
            pop_class = 3
    else:  # num_division == 3
        if popularity < 33:
            pop_class = 0
        elif 33 <= popularity < 66:
            pop_class = 1
        else:
            pop_class = 2
    return pop_class


# List of indexes of the columns needed based on the dataset used
INDEXES = {
    "en": {
        "label": -5,
        "lyrics": -1,
        "release_year": -6,
        "filename": -8,
    },
    "mul": {
        "label": -1,
        "lyrics": -2,
        "release_year": -3,
        "filename": -5,
    }
}


class HSPDataset(Dataset):
    def __init__(self, problem='c', language='en', annotation_file="", augmented=False, subset=None, num_classes=4):
        """
            Builder of a HSPDataset element that inherits from Dataset

            Input:
                - problem: (char) 'c' or 'r' to distinguish between classification and regression so that the target value
                returned will be a popularity score or a class
                - language: (string) 'en' or 'mul' to determine the column indexes of the dataset
                - annotation_file: (string) dataset file path
                - augmented: (boolean) True of False based on the choice of performing data augmentation
                - subset: (Dataframe) object containing the portion of dataset used to create the HSPDataset object
                - num_classes: (int) in case of classification is used to determine the number of classes to build
                the popularity ranges

        """
        self.tracks_dataframe = subset
        self.language = language
        self.indexes = INDEXES[self.language]
        self.root_dir = AUDIO_DIR
        self.problem = problem
        self.num_classes = num_classes
        self.augmented = augmented
        if self.tracks_dataframe is None:
            self.tracks_dataframe = pd.read_csv(annotation_file, index_col=0, encoding='utf8')
        self.tensor_transform = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(128, antialias=True)
        print(len(self.tracks_dataframe))

    def __getitem__(self, index):
        """
            Returns information about the index-th element inside the dataset.

            Input: index (int) of the element to get
            Output: dictionary that contains 4 values regarding {"spectrogram":, "lyrics":, "year":, "label":}
        """
        label = int(self.tracks_dataframe.iloc[index, self.indexes["label"]])
        if self.problem == 'c':
            label = get_class(label, self.num_classes)
        lyrics = self.tracks_dataframe.iloc[index, self.indexes["lyrics"]]
        release_year = int(self.tracks_dataframe.iloc[index, self.indexes["release_year"]])
        filename = self.tracks_dataframe.iloc[index, self.indexes["filename"]]
        file = str(filename) + ".mp3"
        audio_sample_path = os.path.join(self.root_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE,
                                                   hop_length=512, win_length=1024)  # 639450 = 29 sec
        if self.augmented:
            # r = random.randint(0, 1)
            # if r == 0:
            norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            # else:
            #     audio = time_enhancement_channel(audio, SAMPLE_RATE)
            #     norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:639450], sample_rate=SAMPLE_RATE)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        norm_spectrogram = norm_spectrogram[:, :, :1275]  # con hop_size 512 è 1275, con 256 è 2575
        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        result = {"spectrogram": rgb_spectrogram, "lyrics": lyrics, "year": release_year, "label": label}
        return result

    def __len__(self):
        """
            Returns the length of the dataset (int)
        """
        return len(self.tracks_dataframe)
