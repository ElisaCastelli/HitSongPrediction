import librosa as lib
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import librosa.display
import random

audio_sample_path = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/ESC-50-master/audio/1-18755-B-4.wav"


audio, sr = lib.load(audio_sample_path, sr=41000)

mel = lib.feature.melspectrogram(y=np.abs(audio)**2, sr=41000,
                                 hop_length=256,
                                 n_fft=1024, n_mels=128,  power=2.0, fmax=sr/2
                                 )
fig, ax = plt.subplots()
img = librosa.display.specshow(mel, sr=44100)
plt.savefig('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/prove/mel.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)
log_mel = librosa.power_to_db(mel, ref=np.max)
# log_mel = lib.amplitude_to_db(mel, ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(log_mel, sr=44100)
plt.savefig('//Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/prove/log_mel.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)
spectrogram_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
fig, ax = plt.subplots()
img = librosa.display.specshow(spectrogram_normalized, sr=44100)
plt.savefig('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/prove/1-18755-B-4_norm_log_mel.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)

transform = transforms.ToTensor()

image = transform(spectrogram_normalized)

print(image.shape)

resize_crop = transforms.Compose([
            transforms.Resize(size=224, antialias=True),  # Resize the image while maintaining the aspect ratio
            transforms.CenterCrop(224),  # Crop the image to 224x224 pixels from a centeer point
        ])

image = resize_crop(image)

to_image = transforms.ToPILImage()


def load_audio(self):
    waveform = None
    sr = 0
    try:
        waveform, sr = li.load(self.file_path, sr=self.TARGET_SR)
    except Exception as e:
        print(e)
    return waveform[:-1], sr


def resample(self):
    if self.sample_rate != self.TARGET_SR:
        self.waveform = li.resample(self.waveform, orig_sr=self.sample_rate, target_sr=self.TARGET_SR)
        self.sample_rate = self.TARGET_SR
    return self.waveform


def get_image_spectrogram(self):
    file = glob.glob('/nas/home/ecastelli/thesis/Images/' + self.audio_filename[:-4] + '.png')
    if len(file) == 0:
        self.get_save_spectrogram()
    image = mpimg.imread('/nas/home/ecastelli/thesis/Images/' + self.audio_filename[:-4] + '.png')[:, :, :3]
    image = self.resize_crop(image)
    return image


def get_save_spectrogram(self):
    try:
        fig, ax = plt.subplots()
        spec1 = self.get_log_mel_spectrogram(hop_length=256, n_fft=1024)
        img = li.display.specshow(spec1, sr=self.sample_rate)
        plt.savefig('/nas/home/ecastelli/thesis/Images/' + self.audio_filename[:-4] + '.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(self.audio_filename[:-4])



    def get_log_mel_spectrogram(self, waveform, sample_rate=SAMPLE_RATE, hop_length=512, n_fft=1024):
        spectrogram_normalized = None
        try:
            mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft, n_mels=128,
                                                 fmax=8000
                                                 )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            spectrogram_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
        except Exception as e:
            print(e)
        return spectrogram_normalized

    def get_hpss(self, mel_spectrogram):
        H, P = librosa.effects.hpss(mel_spectrogram)
        return H, P


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

