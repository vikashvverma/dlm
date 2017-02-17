# Generate images from celebrities from original looking images
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from dlm.utilities.data_handler import get_imdb_data, split_data, get_data, get_classless_images
import argparse
import logging
import numpy as np
import cv2
import pickle
import os
import time
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

x_list = get_classless_images(path='data/imdb-wiki/handpicked_restructured/Jennifer_Aniston/')
y_list = [[1,0] for _ in range(len(x_list))]

sys.exit(0)

logging.info("Data loaded")

logging.info("Shuffling all the data")
logging.info("Getting numpy RNG state")
rng_state = np.random.get_state()
logging.info("Shuffling X")
np.random.shuffle(x)
logging.info("Setting numpy RNG state, to same as first shuffle")
np.random.set_state(rng_state)
logging.info("Shuffling Y")
np.random.shuffle(y)

logging.info("Classes to arrays")
input_shape = x[0].shape



logging.info("Building model")
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

model.add(Deconvolution2D(nb_filter=3, nb_row=3, nb_col=3, output_shape=(None, 64, 64, 3), subsample=(1,1), border_mode='valid'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])




