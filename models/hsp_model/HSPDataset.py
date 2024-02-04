import os
import torch
import polars as pl
from torch.utils.data import Dataset
from torchvision import transforms
from models.audio_processing import *

SAMPLE_RATE = 22050
# SAMPLE_RATE = 44100


AUDIO_DIR = "/nas/home/ecastelli/thesis/Audio/"


def get_class(popularity, num_division):
    """
        Based on the number of class in which we want to classify the songs popularity,
        taking the popularity value of a song this method returns the associated popularity class
    """
    if popularity==100:
        popularity=99
    unit = 100/num_division
    return int(np.floor(popularity/unit))


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
        if subset:
            self.tracks_dataframe = subset
        else:
            self.tracks_dataframe = pl.read_parquet(annotation_file)
        self.language = language
        self.root_dir = AUDIO_DIR
        self.problem = problem
        self.num_classes = num_classes
        self.augmented = augmented
        self.tensor_transform = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(128, antialias=True)

    def __getitem__(self, index):
        """
            Returns information about the index-th element inside the dataset.

            Input: index (int) of the element to get
            Output: dictionary that contains 4 values regarding {"spectrogram":, "lyrics":, "year":, "label":}
        """
        data_row = self.tracks_dataframe.row(index)
        label = int(data_row['label'])
        if self.problem == 'c':
            label = get_class(label, self.num_classes)
        lyrics = data_row["lyrics"]
        release_year = int(data_row["release_year"])
        filename = data_row["spotify_id"]
        file = str(filename) + ".mp3"
        audio_sample_path = os.path.join(self.root_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)

        # The length is specified because, even if the audios have different lengths, each one is longer than 661500
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

        # 3 channel concat the spectrogram, could be intering to create the 3 channels with an harmonic percussive separation

        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        result = {"spectrogram": rgb_spectrogram, "lyrics": lyrics, "year": release_year, "label": label}
        return result

    def __len__(self):
        """
            Returns the length of the dataset (int)
        """
        return len(self.tracks_dataframe)


