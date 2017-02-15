from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from utilities.data_handler import get_imdb_data, split_data, get_data
from collections import OrderedDict
import argparse
import logging
import numpy as np
import cv2
import pickle
import os
import time
import json
import sys

np.random.seed(1337)

parser = argparse.ArgumentParser(description="Generate images with neural networks")
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
args = parser.parse_args()
loglevel = logging.ERROR

if args.verbosity == 1:
	loglevel = logging.WARNING
elif args.verbosity == 2:
	loglevel = logging.INFO
elif args.verbosity == 3:
	loglevel = logging.DEBUG

logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)

# Two classes because classification only specifies if the image is generated or real
nb_classes = 2
logging.info("Building model")
"""
MODEL HERE
"""

model = Sequential()
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same',input_shape=input_shape,subsample=(1,1)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))

model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
#model.add(Activation('relu'))
#model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
#model.add(Activation('relu'))
#model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
#model.add(Activation('relu'))

model.add(Dropout(0.1))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
