#import matplotlib.pyplot as plt
#import seabprn as sns
#from IPython import display
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
from keras.api._v2.keras.layers.experimental import preprocessing
from keras import layers
from keras import models
import tensorflow_datasets as tfds

class SpeechMusicClassifier:
    def __init__(self):
        #self.ds = tfds.load('GTZANMusicSpeech', split='train', shuffle_files = True)
        self.data_dir = "c:\SpeechMusic"
        self.categories = self.set_categories()
        self.num_labels = len(self.categories)
        self.filenames = self.set_filenames()
        print(len(self.filenames))
        self.TRAIN_SIZE = 0.75
        self.train_files, self.val_files = self.train_val_file()
        print(len(self.train_files))
        print(len(self.val_files))
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.file_ds = tf.data.Dataset.from_tensor_slices(self.train_files)
        self.waveform_ds = self.file_ds.map(self.get_waveform_and_label, num_parallel_calls = self.AUTOTUNE)
        self.spectrogram_ds = self.waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls = self.AUTOTUNE)
        #self.train_ds = preprocess_dataset(self.train_files)
        #self.val_ds = preprocess_dataset(self.val_files)
        self.train_ds, self.val_ds = self.prepare_train_and_val_ds()
        self.batch_size = 32
        self.model = self.prepare_model()


    def set_categories(self):
        categories = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        categories = [category for category in categories if 'wav' in category]
        return categories
    
    def set_filenames(self):
        filenames = tf.io.gfile.glob(str(self.data_dir)+ '/*/*')
        filenames = [filename for filename in filenames if 'wav' in filename]
        filenames = tf.random.shuffle(filenames)
        return filenames

    def train_val_file(self):
        train_files = self.filenames[:int(len(self.filenames)*self.TRAIN_SIZE)]
        val_files = self.filenames[int(len(self.filenames)*self.TRAIN_SIZE):]
        return train_files, val_files

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2]

    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        #waveform = decode_audio(audio_binary)
        return audio_binary, label

    def get_spectrogram(self,waveform):
        print(waveform.shape)
        spectrogram = tf.signal.stft(signals=waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    def get_spectrogram_and_label_id(self, audio, label):
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label==self.categories)
        return spectrogram, label_id

    def preprocess_dataset(self,files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls = self.AUTOTUNE)
        output_ds = output_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=self.AUTOTUNE)
        return output_ds

    def prepare_train_and_val_ds(self):
        train_ds = self.preprocess_dataset(self.train_files)
        val_ds = self.preprocess_dataset(self.val_files)
        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)
        train_ds = train_ds.cache().prefetch(self.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(self.AUTOTUNE)
        return train_ds, val_ds

    def prepare_model(self):
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(self.spectrogram_ds.map(lambda x, _: x))
        for spectrogram, _ in self.spectrogram_ds.take(1):
            input_shape = spectrogram.shape

        model = model.Sequential([
            layers.Input(shape = input_shape),
            preprocessing.Resizing(64, 64),
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_labels),
        ])

        model.compile(
            optimizer = tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrosssentropy(from_logits = True),
            metrics=['accuracy'],
        )

        EPOCHS = 10
        history = model.fit(
            self.train_ds,
            validation_data = self.val_ds,
            epochs = EPOCHS,
            callbacks = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        return model




classif = SpeechMusicClassifier()
model = classif.model
model.summary()