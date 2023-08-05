import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import librosa
import librosa.display
from ESCAudioPreProcessing import AudioPreProcessing

ANNOTATIONS_FILE = "/nas/home/ecastelli/thesis/ESC-50-master/meta/esc50.csv"
AUDIO_DIR = "/nas/home/ecastelli/thesis/ESC-50-master/audio"
# ANNOTATIONS_FILE = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/ESC-50-master/meta/esc50.csv"
# AUDIO_DIR = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/ESC-50-master/audio"
SAMPLE_RATE = 44100
NUM_SAMPLES = 220500


class ESC50Dataset(Dataset):

    def __init__(self, subset,
                 annotations_file=ANNOTATIONS_FILE,
                 audio_dir=AUDIO_DIR,
                 augmented=False,
                 ):
        self.audio_process = AudioPreProcessing()
        self.annotations = pd.read_csv(annotations_file)
        if subset is not None:
            self.annotations = subset
        self.audio_dir = audio_dir
        self.tensor_transf = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from a centeer point
        ])
        self.target_sample_rate = SAMPLE_RATE
        self.num_samples = NUM_SAMPLES
        self.augmented = augmented

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label = self._get_audio_sample_label(index)
        audio_sample_path = self._get_audio_sample_path(index)
        audio, sr = librosa.load(audio_sample_path, sr=SAMPLE_RATE)
        if self.augmented is True:
            audio = self.audio_process.time_enhancement_channel(waveform=audio)
        normalized_spectrogram = self.audio_process.get_log_mel_spectrogram(audio, sample_rate=SAMPLE_RATE,
                                                              hop_length=512, n_fft=1024)
        harmonic, percussive = self.audio_process.get_hpss(normalized_spectrogram)
        if self.augmented is True:
            normalized_spectrogram = self.audio_process.spec_enhancement_channel(mel_spectrogram=normalized_spectrogram)
            harmonic = self.audio_process.spec_enhancement_channel(mel_spectrogram=harmonic)
            percussive = self.audio_process.spec_enhancement_channel(mel_spectrogram=percussive)
        harmonic = self.audio_process.tensor_transf(harmonic)
        percussive = self.audio_process.tensor_transf(percussive)
        normalized_spectrogram = self.audio_process.tensor_transf(normalized_spectrogram)
        harmonic = self.audio_process.resize_crop(harmonic.real)
        percussive = self.audio_process.resize_crop(percussive.real)
        normalized_spectrogram = self.audio_process.resize_crop(normalized_spectrogram)
        rgb_spectrogram = torch.concat([harmonic, normalized_spectrogram, percussive], axis=0)
        return rgb_spectrogram, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]

    def read_csv_as_dict(self):
        data = self.annotations.to_dict('records')
        fold_data = [[] for _ in range(6)]  # Create empty lists for each fold
        for row in data:
            fold_index = int(row['fold'])
            fold_data[fold_index].append(row)
        return fold_data[1:]

#
# if __name__ == '__main__':
#     esc = ESC50Dataset(None, augmented=True)
#     esc[1]
