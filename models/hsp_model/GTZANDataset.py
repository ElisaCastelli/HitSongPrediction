import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import librosa.feature
import librosa.display
from models.hsp_model.audio_processing import *


ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/GTZAN/features_30_sec.csv"
AUDIO_DIR = "/nas/home/ecastelli/thesis/GTZAN/genres_original"

SAMPLE_RATE = 22050

''' List of genres present in GTZANGenre Dataset'''
DICT_LABEL = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9,
}


class GTZANDataset(Dataset):
    def __init__(self, subset, augmented=False):
        if subset is None:
            subset = pd.read_csv(ANNOTATIONS_FILE)
        self.annotations = subset
        self.audio_dir = AUDIO_DIR
        self.tensor_transform = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(128, antialias=True)
        self.augmented = augmented

    def __getitem__(self, index):
        """
            Returns the melspectrogram and the popularity target value of the index-th element inside the dataset
        """
        label = self.annotations.iloc[index, -1]
        filename = self.annotations.iloc[index, 0]
        file = str(label) + "/" + str(filename)
        audio_sample_path = os.path.join(self.audio_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE,
                                                   hop_length=512, win_length=1024)
        if self.augmented:
            r = random.randint(0, 1)
            if r == 0:
                norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            else:
                audio = time_enhancement_channel(audio, sample_rate=SAMPLE_RATE)
                norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE,
                                                           hop_length=512, win_length=1024)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        norm_spectrogram = norm_spectrogram[:, :, :1275]
        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        target = DICT_LABEL[label]
        return rgb_spectrogram, target

    def __len__(self):
        """
            Returns the length of the dataset
        """
        return len(self.annotations)
