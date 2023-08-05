import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import librosa
import librosa.feature
from torchvision import transforms
import librosa.display
from torchvision.transforms.functional import crop
import random

SAMPLE_RATE = 16000
AUDIO_DIR = "/nas/home/ecastelli/thesis/Billboard/AudioBB/"
ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_FINAL.csv"


def get_log_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, hop_length=512, n_fft=2048):
    spectrogram_normalized = None
    try:
        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate,
                                             hop_length=hop_length, n_fft=n_fft,
                                             n_mels=256, fmax=8000
                                             )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        spectrogram_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    except Exception as e:
        print(e)
    return spectrogram_normalized


def crop_mel(mel_spectrogram):
    crop1 = crop(mel_spectrogram, top=0, left=0, height=224, width=224)
    crop2 = crop(mel_spectrogram, top=0, left=236, height=224, width=224)
    crop3 = crop(mel_spectrogram, top=0, left=480, height=224, width=224)
    return crop1, crop2, crop3


def spec_enhancement_channel(mel_spectrogram, frequency_masking_para=16,
                             time_masking_para=75, frequency_mask_num=1, time_mask_num=2):
    # mel_spectrogram = mel_spectrogram.squeeze(0)
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        # f = torch.randint(0, frequency_masking_para, (1,))
        f = int(f)
        # f0 = torch.randint(0, v-f, (1,))
        f0 = random.randint(0, v - f)
        mel_spectrogram[f0:f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        # t = torch.randint(0, time_masking_para, (1,))
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        # t0 = torch.randint(0, tau-t, (1,))
        mel_spectrogram[:, t0:t0 + t] = 0

    return mel_spectrogram


class BBDataset(Dataset):
    def __init__(self, problem='regression', augmented=False, subset=None):
        self.tracks_dataframe = subset
        if self.tracks_dataframe is None:
            self.tracks_dataframe = pd.read_csv(ANNOTATIONS_FILE, index_col=1)
        self.root_dir = AUDIO_DIR
        self.problem = problem
        self.augmented = augmented
        self.tensor_transform = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(224, antialias=True)

    def __getitem__(self, index):
        if self.problem == 'classification':
            label = self.tracks_dataframe.iloc[index, -1]
        else:
            label = self.tracks_dataframe.iloc[index, -5]
        lyrics = self.tracks_dataframe.iloc[index, -3]
        release_year = self.tracks_dataframe.iloc[index, -2]
        filename = self.tracks_dataframe.iloc[index, -6]
        file = str(label) + "/" + str(filename)
        audio_sample_path = os.path.join(self.root_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio, sample_rate=SAMPLE_RATE)
        if self.augmented:
            norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            # norm_spectrogram = norm_spectrogram.unsqueeze(0)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        crop1, crop2, crop3 = crop_mel(norm_spectrogram.cuda())
        rgb_spectrogram = torch.cat((crop1.cuda(), crop2.cuda(), crop3.cuda()), 0)
        return rgb_spectrogram, label

    def __len__(self):
        return len(self.tracks_dataframe)
