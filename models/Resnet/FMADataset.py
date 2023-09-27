import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
import librosa
import librosa.feature
import librosa.display
from torchvision.transforms.functional import crop
import random

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/FMA/tracks.csv"
GENRE_ANNOTATION = "/nas/home/ecastelli/thesis/FMA/genres.csv"
AUDIO_DIR = "/nas/home/ecastelli/thesis/FMA/fma_medium"

SAMPLE_RATE = 22050

# album                 comments
# album.1           date_created
# album.2          date_released
# album.3               engineer
# album.4              favorites
# album.5                     id
# album.6            information
# album.7                listens
# album.8               producer
# album.9                   tags
# album.10                 title
# album.11                tracks
# album.12                  type
# artist       active_year_begin
# artist.1       active_year_end
# artist.2     associated_labels
# artist.3                   bio
# artist.4              comments
# artist.5          date_created
# artist.6             favorites
# artist.7                    id
# artist.8              latitude
# artist.9              location
# artist.10            longitude
# artist.11              members
# artist.12                 name
# artist.13     related_projects
# artist.14                 tags
# artist.15              website
# artist.16       wikipedia_page
# set                      split --> training, validation, test
# set.1                   subset --> medium, small, full
# track                 bit_rate
# track.1               comments
# track.2               composer
# track.3           date_created
# track.4          date_recorded
# track.5               duration
# track.6              favorites
# track.7              genre_top -->
# track.8                 genres
# track.9             genres_all
# track.10           information
# track.11              interest
# track.12         language_code
# track.13               license
# track.14               listens
# track.15              lyricist
# track.16                number
# track.17             publisher
# track.18                  tags
# track.19                 title
genres_map = {
    'Blues': 0,
    'Classical': 1,
    'Country': 2,
    'Easy Listening': 3,
    'Electronic': 4,
    'Experimental': 5,
    'Folk': 6,
    'Hip-Hop': 7,
    'Instrumental': 8,
    'International': 9,
    'Jazz': 10,
    'Old-Time / Historic': 11,
    'Pop': 12,
    'Rock': 13,
    'Soul-RnB': 14,
    'Spoken': 15,
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
    y = librosa.util.fix_length(data=y, size=639450)
    return y


class FMADataset(Dataset):
    def __init__(self, subset, augmented=False):
        if subset is None:
            self.medium_dataset = pd.read_csv(ANNOTATIONS_FILE, encoding='utf')
        else:
            self.medium_dataset = subset
        self.augmented = augmented
        # self.medium_dataset = self.tracks_dataframe[self.tracks_dataframe['set.1'] == "medium"]
        # self.small_dataset = self.tracks_dataframe[self.tracks_dataframe['set.1'] == "small"]
        self.genre_dataframe = pd.read_csv(GENRE_ANNOTATION, encoding='utf')
        self.tensor_transform = transforms.ToTensor()

    def get_path(self, name_file):
        zero_to_add = 6 - len(str(name_file))
        for i in range(0, zero_to_add):
            name_file = '0' + str(name_file)
        path = AUDIO_DIR + "/" + str(name_file)[0:3] + "/" + str(name_file) + ".mp3"
        return path

    def get_genre_id(self, genre):
        genre_id = genres_map[genre]
        return genre_id

    def __getitem__(self, index):
        name_file = self.medium_dataset.iloc[index][0]
        path = self.get_path(name_file)
        genre = self.medium_dataset.iloc[index]['track.7']
        genre_id = self.get_genre_id(genre)
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)
        norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:639450],
                                           sample_rate=SAMPLE_RATE)  # prendo 29 secondi per essere sicura --> 256x1249
        if self.augmented:
            r = random.randint(0, 1)
            if r == 0:
                norm_spectrogram = spec_enhancement_channel(norm_spectrogram)
            else:
                audio = time_enhancement_channel(audio)
                norm_spectrogram = get_log_mel_spectrogram(waveform=audio[:639450], sample_rate=SAMPLE_RATE)
        norm_spectrogram = self.tensor_transform(norm_spectrogram)
        norm_spectrogram = norm_spectrogram[:, :, :1245]
        rgb_spectrogram = torch.cat((norm_spectrogram, norm_spectrogram, norm_spectrogram), 0)
        return rgb_spectrogram, genre_id

    def __len__(self):
        return len(self.medium_dataset)


if __name__ == '__main__':
    fma = FMADataset(subset=None, augmented=True)
    print(fma[3])
