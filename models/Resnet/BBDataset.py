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

SAMPLE_RATE = 22050
# SAMPLE_RATE = 44100


AUDIO_DIR = "/nas/home/ecastelli/thesis/Audio/"
# ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/Billboard/CSV/SPD_all_lang_not_updated.csv" # 24645 songs 32ksongs
ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/Billboard/CSV/SPD_en_no_dup.csv"


def get_class(popularity):
    if popularity < 25:
        pop_class = 0
    elif 25 <= popularity < 50:
        pop_class = 1
    elif 50 <= popularity < 75:
        pop_class = 2
    else:
        pop_class = 3
    return pop_class


def get_hpss(y, hop_length=128, win_length=2048):
    D = np.abs(librosa.stft(y=y, win_length=win_length, hop_length=hop_length, n_fft=win_length))
    h, p = librosa.decompose.hpss(D)
    h = librosa.power_to_db(h, ref=np.max)
    p = librosa.power_to_db(p, ref=np.max)
    h = (h - h.min()) / (h.max() - h.min())
    p = (p - p.min()) / (p.max() - p.min())
    return h, p


def get_log_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, hop_length=512, win_length=1024):
    spectrogram_normalized = None
    try:
        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, win_length=win_length,
                                             hop_length=hop_length, n_fft=win_length,
                                             n_mels=256,
                                             fmax=8000
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
    y_trimmed, index = librosa.effects.trim(y=waveform, top_db=40, frame_length=1024, hop_length=512)
    rate = random.uniform(0.5, 1.5)
    y = librosa.effects.time_stretch(y=y_trimmed, rate=rate)  # rate from 0.5 to 1.5 randomly
    steps = random.randint(1, 5)
    y = librosa.effects.pitch_shift(y, n_steps=steps, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(data=y, size=639450)
    return y


class BBDataset(Dataset):
    def __init__(self, problem='c', augmented=False, subset=None):
        self.tracks_dataframe = subset
        if self.tracks_dataframe is None:
            self.tracks_dataframe = pd.read_csv(ANNOTATIONS_FILE, index_col=0, encoding='utf8')
        self.root_dir = AUDIO_DIR
        self.problem = problem
        self.augmented = augmented
        self.tensor_transform = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(128, antialias=True)
        print(len(self.tracks_dataframe))

    def __getitem__(self, index):
        label = int(self.tracks_dataframe.iloc[index, -5])  # SPD ALL LANG -1 SPD EN -5
        if self.problem == 'c':
            label = get_class(label)
        lyrics = self.tracks_dataframe.iloc[index, -1]  # SPD ALL LANG -2 SPD EN -1
        release_year = int(self.tracks_dataframe.iloc[index, -6])  # SPD ALL LANG -3 SPD EN -6
        filename = self.tracks_dataframe.iloc[index, -8]  # SPD ALL LANG -5 SPD EN -8
        file = str(filename) + ".mp3"
        audio_sample_path = os.path.join(self.root_dir, file)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:661500], sample_rate=SAMPLE_RATE)  # 639450 = 29 sec
        if self.augmented:
            r = random.randint(0, 1)
            #if r == 0:
            norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            #else:
            #     audio = time_enhancement_channel(audio)
            #     norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:639450], sample_rate=SAMPLE_RATE)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        norm_spectrogram = norm_spectrogram[:, :, :1275]
        # print(norm_spectrogram.shape)# con hop_size 512 è 1275, con 256 è 2575
        # h, p = get_hpss(audio[:639450])
        # h = self.tensor_transform(h)
        # h = self.resize_mel(h)
        # p = self.tensor_transform(p)
        # p = self.resize_mel(p)
        # h = h[:, :, :1245]
        # p = p[:, :, :1245]
        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        result = {"spectrogram": rgb_spectrogram, "lyrics": lyrics, "year": release_year,  "label": label}  # "lyrics": lyrics,
        return result

    def __len__(self):
        return len(self.tracks_dataframe)
#

if __name__ == '__main__':
    b = BBDataset('c')
    b[55]
