import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import librosa
from AudioPreProcessing import AudioPreProcessing
import librosa.display
import matplotlib.pyplot as plt
import numpy

SAMPLE_RATE = 16000
AUDIO_DIR = "/nas/home/ecastelli/thesis/Audio/"
ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/input/df_final_english.csv"


class TracksDataset(Dataset):
    def __init__(self, problem='regression', augmented=False):
        self.tracks_dataframe = pd.read_csv(ANNOTATIONS_FILE, index_col=1)
        self.root_dir = AUDIO_DIR
        self.problem = problem
        self.augmented = augmented
        self.audio_process = AudioPreProcessing()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        track_id = self.tracks_dataframe.iloc[idx, 2]
        audio_sample_path = os.path.join(self.root_dir, str(track_id) + ".mp3")
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        #audio = librosa.util.fix_length(data=audio, size=661500)
        if self.augmented is True:
            audio = self.audio_process.time_enhancement_channel(waveform=audio)
        normalized_spectrogram = self.audio_process.get_log_mel_spectrogram(audio, sample_rate=SAMPLE_RATE,
                                                                            hop_length=512, n_fft=2048)
        normalized_spectrogram = self.audio_process.tensor_transf(normalized_spectrogram)
        normalized_spectrogram = self.audio_process.resize_mel(normalized_spectrogram)
        crop1, crop2, crop3 = self.audio_process.crop_mel(normalized_spectrogram)
        rgb_spectrogram = torch.cat((crop1, crop2, crop3), 0)
        # lyrics = self.tracks_dataframe.iloc[idx, 3]
        year = self.tracks_dataframe.iloc[idx, 5]
        popularity = self.tracks_dataframe.iloc[idx, 4]
        return {'image': rgb_spectrogram, 'year': year, 'popularity': popularity}

    def __len__(self):
        return len(self.tracks_dataframe)

    def save_image(self, log_mel_spectrogram, sr, hop_length, idx, fig_type):
        if torch.is_tensor(log_mel_spectrogram):
            log_mel_spectrogram = log_mel_spectrogram.squeeze()
            log_mel_spectrogram = log_mel_spectrogram.numpy()
        if fig_type == "crop1" or fig_type == "crop2" or fig_type == "crop3":
            plt.figure(figsize=(4, 4))
        else:
            plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')
        plt.title('Log Mel Spectrogram')

        # Remove the margins
        plt.tight_layout(pad=0)

        # Save the displayed figure as a PNG file
        output_path = f'/nas/home/ecastelli/thesis/images/{idx}_{fig_type}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

        # Show the plot (optional)
        plt.show()


# harmonic, percussive = self.audio_process.get_hpss(normalized_spectrogram)
# if self.augmented is True:
#     normalized_spectrogram = self.audio_process.spec_enhancement_channel(mel_spectrogram=normalized_spectrogram)
#     harmonic = self.audio_process.spec_enhancement_channel(mel_spectrogram=harmonic)
#     percussive = self.audio_process.spec_enhancement_channel(mel_spectrogram=percussive)
# harmonic = self.audio_process.tensor_transf(harmonic)
# percussive = self.audio_process.tensor_transf(percussive)
# harmonic = self.audio_process.resize_crop(harmonic.real)
# harm_np = harmonic.squeeze(0)
# percussive = self.audio_process.resize_crop(percussive.real)
