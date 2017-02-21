# Generate images from celebrities from original looking images
from dlm.utilities.data_handler import get_imdb_data, split_data, get_data, get_classless_images
import argparse
import logging
import numpy as np
import cv2
import pickle
import os
import time
import sys
from keras.utils import np_utils
from tqdm import tqdm
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.models import Model
import random

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

#x_list = get_classless_images(path='data/imdb-wiki/handpicked_restructured/Jennifer_Aniston/')
#y_list = [[1,0] for _ in range(len(x_list))]

x,y = get_data(path='data/imdb-wiki/handpicked_restructured/')
# Because floats improves learning efficiency according to http://datascience.stackexchange.com/questions/13636/neural-network-data-type-conversion-float-from-int
x /= 255


logging.info("Data loaded")

logging.info("Classes to arrays")
input_shape = x[0].shape
classes = set(y)
nb_classes = len(classes)

logging.info("Categorizes output data")
uniques, ids = np.unique(y, return_inverse=True)
cat_y = np_utils.to_categorical(ids, len(uniques))
(x_train, x_test), (y_train, y_test) = split_data(x, cat_y)

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build the generator model
g_input = Input(shape=[100])
H = Dense(200*32*32)(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape( [200, 32, 32] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(2, 3, 3, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(4, 3, 3, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(1, 1, 1, border_mode='same')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)


d_input = Input(shape=(64,64,3))
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='binary_crossentropy', optimizer=dopt)


def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val
	
make_trainable(discriminator, False)



# Build stacked GAN model
"""
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()
"""

def save_gen(n_ex=16):
	noise = np.random.uniform(0,1,size=[n_ex, 100])
	generated_images = generator.predict(noise)
	
	for index,img in enumerate(generated_images):
		imgpath = "{}/{}_{:06d}.jpg".format(save_path, imgname, index)
		cv2.imwrite(imgpath,img)

ntrain = 10000

#trainidx = random.sample(range(0,x_train.shape[0]), ntrain)
#XT = x_train[trainidx,:,:,:]








