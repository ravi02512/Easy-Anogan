from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math
import keras


from keras.utils. generic_utils import Progbar

def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128*224*224)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((224, 224, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### discriminator model define
def discriminator_model():
    model = keras.applications.resnet50.ResNet50()
    model.layers.pop()
    model.layers.pop()
    inp=model.layers[-1].output
    fc1 = Flatten()(inp)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    
    model = Model(inputs=[model.layers[0].input], outputs=[outputs])
    return model



##    for layer in model.layers:
##        layer.trainable=False
##    last = model.layers[-1].output
##    x = Dense(len(classes), activation="softmax", name='final_output_bolt')(last)
##    finetuned_model = Model(model.input, x)
##    finetuned_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



d=generator_model()
d.summary()
























