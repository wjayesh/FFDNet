import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses 
from tensorflow.keras.layers import Input, Add, subtract, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
import pandas as pd

frmo noise_map import get_noise_map
from sampling import upsample, downsample

tf.config.run_functions_eagerly(True)

# define an FFDNet model
class FFDNet(Model):      
  def __init__(self):
    super(FFDNet, self).__init__()
    self.depth = 4
    self.feature_map = 8
    self.dncnn = tf.keras.Sequential()
    self.dncnn.add(layers.Input(shape=(5, 25, 9)))
    self.dncnn.add(layers.Conv2D(self.feature_map, (3, 3), activation='relu', padding='same', input_shape=(5, 25, 9)))
    for i in range(self.depth-2):
        self.dncnn.add(layers.Conv2D(self.feature_map, (3, 3), padding='same'))
        self.dncnn.add(layers.BatchNormalization())
        self.dncnn.add(layers.Activation('relu'))
    self.dncnn.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))

  def call(self, x):
    x = downsample(x)
    noise_map = get_noise_map(x)
    x = np.concatenate((noise_map, x), axis=3);
    cleaned = self.dncnn(x)
    output = upsample(cleaned)
    return cleaned
