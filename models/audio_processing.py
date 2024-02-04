import numpy as np
import librosa
import random
from torchvision.transforms.functional import crop


def get_hpss(y, hop_length=128, win_length=2048):
    D = np.abs(librosa.stft(y=y, win_length=win_length, hop_length=hop_length, n_fft=win_length))
    h, p = librosa.decompose.hpss(D)
    h = librosa.power_to_db(h, ref=np.max)
    p = librosa.power_to_db(p, ref=np.max)
    h = (h - h.min()) / (h.max() - h.min())
    p = (p - p.min()) / (p.max() - p.min())
    return h, p


def get_log_mel_spectrogram(waveform, sample_rate=22050, hop_length=512, win_length=1024):
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

    # Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        # f = torch.randint(0, frequency_masking_para, (1,))
        f = int(f)
        # f0 = torch.randint(0, v-f, (1,))
        f0 = random.randint(0, v - f)
        mel_spectrogram[f0:f0 + f, :] = 0

    # Time masking
    for i in range(time_mask_num):
        # t = torch.randint(0, time_masking_para, (1,))
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        # t0 = torch.randint(0, tau-t, (1,))
        mel_spectrogram[:, t0:t0 + t] = 0

    return mel_spectrogram


def time_enhancement_channel(waveform, sample_rate):
    y_trimmed, index = librosa.effects.trim(y=waveform, top_db=40, frame_length=1024, hop_length=512)
    rate = random.uniform(0.5, 1.5)
    y = librosa.effects.time_stretch(y=y_trimmed, rate=rate)  # rate from 0.5 to 1.5 randomly
    steps = random.randint(1, 5)
    y = librosa.effects.pitch_shift(y, n_steps=steps, sr=sample_rate)
    y = librosa.util.fix_length(data=y, size=639450)
    return y
