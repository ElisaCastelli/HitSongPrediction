import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
import librosa
import librosa.feature
import librosa.display
import random

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

def get_log_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, hop_length=512, win_length=1024):
    spectrogram_normalized = None
    try:
        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate,
                                             hop_length=hop_length, win_length=win_length,
                                             n_mels=256, fmax=8000, n_fft=win_length
                                             )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        spectrogram_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    except Exception as e:
        print(e)
    return spectrogram_normalized


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


def time_enhancement_channel(waveform):
    y_trimmed, index = librosa.effects.trim(y=waveform, top_db=40, frame_length=2048, hop_length=1024)
    rate = random.uniform(0.5, 1.5)
    y = librosa.effects.time_stretch(y=y_trimmed, rate=rate)  # rate from 0.5 to 1.5 randomly
    steps = random.randint(1, 5)
    y = librosa.effects.pitch_shift(y, n_steps=steps, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(data=y, size=661500)
    return y


def get_hpss(y, hop_length=256, win_length=1024):
    D = np.abs(librosa.stft(y=y, win_length=win_length, hop_length=hop_length, n_fft=win_length))
    h, p = librosa.decompose.hpss(D)
    h = librosa.power_to_db(h, ref=np.max)
    p = librosa.power_to_db(p, ref=np.max)
    h = (h - h.min()) / (h.max() - h.min())
    p = (p - p.min()) / (p.max() - p.min())
    return h, p


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
        label = self.annotations.iloc[index, -1]
        filename = self.annotations.iloc[index, 0]
        file = str(label) + "/" + str(filename)
        audio_sample_path = os.path.join(self.audio_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE)
        if self.augmented:
            r = random.randint(0, 1)
            if r == 0:
                norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            else:
                audio = time_enhancement_channel(audio)
                norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        norm_spectrogram = norm_spectrogram[:, :, :1275]
        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        target = DICT_LABEL[label]
        return rgb_spectrogram, target

    def __len__(self):
        return len(self.annotations)
