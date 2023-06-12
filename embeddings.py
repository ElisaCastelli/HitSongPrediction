import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models, losses, Model
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

class EmbeddingExtractor:
    def __init__(self, mel_vector):
        self.mel_vector = mel_vector
        self.model = None
        self.pretrained_model = self.pretrained_model_setup()

    def pretrained_model_setup(self, height, width, channel):
        self.pretrained_model= tf.keras.applications.ResNet50(include_top=False, #include_top false is useful to not take also the fully connected layer part
                           input_shape=(height,width,channel),
                           pooling='avg',
                           weights='imagenet')
        for layer in self.pretrained_model.layers:
                layer.trainable=False
        self.pretrained_model.summary()
        
    def create_model(self):
        input_shape = self.mel_vector.shape[1,2] # dimensioni dei mel
        self.model = tf.keras.Sequential([
          self.pretrained_model, # rivedi i livelli aggiuntivi
          tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),                     
          tf.keras.layers.Dropout(0.2),                 
          tf.keras.layers.GlobalAveragePooling2D(),                 
          tf.keras.layers.Dense(units=100, activation='softmax')      #erano 5 classi in output ma io voglio regressione            
        ])                  

    def configure_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
        self.model.summary()
        print('Number of trainable weights = {}'.format(len(self.model.trainable_weights)))
    
    