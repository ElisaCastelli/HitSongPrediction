import librosa as li
import librosa.feature
import numpy as np
import librosa
from torchvision import transforms
import glob
import random
from torchvision import transforms
from torchvision.transforms.functional import crop

AUDIO_DIR = "/nas/home/ecastelli/thesis/Audio/"
# AUDIO_DIR = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Audio/"
SAMPLE_RATE = 16000


class AudioPreProcessing:
    def __init__(self):
        self.tensor_transf = transforms.ToTensor()
        self.resize_crop = transforms.Compose([
            transforms.Resize(256, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from the center
        ])
        self.resize_mel = transforms.Resize(224, antialias=True)

    def get_log_mel_spectrogram(self, waveform, sample_rate=SAMPLE_RATE, hop_length=512, n_fft=1024):
        spectrogram_normalized = None
        try:
            mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate,
                                                 hop_length=hop_length, n_fft=n_fft,
                                                 n_mels=256, fmax = 8000
                                                 )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            spectrogram_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
        except Exception as e:
            print(e)
        return spectrogram_normalized

    def get_hpss(self, mel_spectrogram):
        h, p = li.effects.hpss(mel_spectrogram)
        return h, p

    def time_enhancement_channel(self, waveform):
        y_trimmed, index = librosa.effects.trim(y=waveform, top_db=40, frame_length=1024, hop_length=512)
        rate = random.uniform(0.5, 1.5)
        y = librosa.effects.time_stretch(y=y_trimmed, rate=rate)  # rate from 0.5 to 1.5 randomly
        steps = random.randint(1, 5)
        y = librosa.effects.pitch_shift(y, n_steps=steps, sr=44100)
        y = librosa.util.fix_length(data=y, size=220500)
        return y

    def spec_enhancement_channel(self, mel_spectrogram, frequency_masking_para=16,
                                 time_masking_para=75, frequency_mask_num=1, time_mask_num=2):
        v = mel_spectrogram.shape[0]
        tau = mel_spectrogram.shape[1]

        # Step 2 : Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v - f)
            mel_spectrogram[f0:f0 + f, :] = 0

        # Step 3 : Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau - t)
            mel_spectrogram[:, t0:t0 + t] = 0

        return mel_spectrogram

    def crop_mel(self, mel_spectrogram):
        crop1 = crop(mel_spectrogram, top=0, left=0, height=224, width=224)
        crop2 = crop(mel_spectrogram, top=0, left=224, height=224, width=224)
        crop3 = crop(mel_spectrogram, top=0, left=448, height=224, width=224)
        return crop1, crop2, crop3
