# Generate images from celebrities from original looking images
from dlm.utilities.data_handler import get_imdb_data, split_data, get_data, get_classless_images, equal_shuffle
import argparse
import logging
import math
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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.models import Model
import random

# Reproducibility 
np.random.seed(2000)
random.seed(2000)

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

gan_data_path = "data/gan_data/models"
date_str = time.strftime("%d-%m-%Y_%H-%M-%S")
save_dir = "{}/{}".format(gan_data_path, date_str)
os.makedirs(save_dir, exist_ok=True)

def create_collage(images, path='collage.jpg'):
	N = len(images)
	rows = int(math.floor(math.sqrt(N)))
	cols = int(N / rows)
	# Add one column if N is not perfect square
	if N != rows*cols:
		cols += 1
	# Add blanks if too many slots
	for _ in range(rows*cols-N):
		blank = np.zeros(shape=images.shape[1:])
		images = np.append(images, [blank], axis=0)
	image_index = 0
	collage = None
	for r in range(rows):
		col_img = None
		for c in range(cols):
			if col_img is None:
				col_img = images[image_index]
			else:
				col_img = np.hstack([col_img, images[image_index]])
			image_index += 1

		if collage is None:
			collage = col_img
		else:
			collage = np.vstack([collage, col_img])
	cv2.imwrite(path, collage)
				


x,y = get_data(path='data/imdb-wiki/handpicked_restructured/')
# Because floats improves learning efficiency according to http://datascience.stackexchange.com/questions/13636/neural-network-data-type-conversion-float-from-int
x /= 255

sx, sy = [],[]
selected_class = "Jennifer Aniston"
logging.info("Selecting class {}".format(selected_class))
for xe,ye in zip(x,y):
	if ye == selected_class:
		sx.append(xe)
		sy.append(ye)
		

sx = np.array(sx)
sy = np.array(sy)

logging.info("Data loaded")

logging.info("Classes to arrays")
input_shape = sx[0].shape
classes = set(sy)
nb_classes = len(classes)

logging.info("Categorizes output data")
uniques, ids = np.unique(sy, return_inverse=True)
cat_y = np_utils.to_categorical(ids, len(uniques))
(x_train, x_test), (y_train, y_test) = split_data(sx, cat_y, ratio=0.90)

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)


# Build the generator model
g_input = Input(shape=[100])
H = Dense(int(3*64*64))(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape( [64, 64, 3] )(H)
H = Convolution2D(30, 3, 3, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(60, 3, 3, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(120, 5, 5, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(240, 5, 5, border_mode='same')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Deconvolution2D(3, 3, 3, border_mode='same', output_shape=(None, 64,64, 3))(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
print("Generator output: {}".format(generator.output_shape))


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
print("Discriminator output: {}".format(discriminator.output_shape))



def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

	
make_trainable(discriminator, False)
# Build stacked GAN model

gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
# GAN.summary()

def save_gen(number=16):
	noise = np.random.uniform(0,1,size=[number, 100])
	generated_images = generator.predict(noise)
	
	for index,img in enumerate(generated_images):
		imgpath = "{}/{}_{:06d}.jpg".format(save_path, imgname, index)
		cv2.imwrite(imgpath,img)


ntrain = 230

trainidx = random.sample(range(0,len(x_train)), ntrain)
XT = x_train[trainidx,:,:,:]


# Pretrain discriminator
noise = np.random.uniform(0,1, size=[len(XT), 100])
generated_images = generator.predict(noise)
X = np.concatenate((XT, generated_images))
n = len(XT)
y = np.zeros(shape=(2*n,2))

# Set class values (real or generated image)
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator, True)
discriminator.fit(X,y, nb_epoch=3, batch_size=10)
y_hat = discriminator.predict(X)


# Measure accuracy of pre-trained discriminator
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx-y_hat_idx
n_total = len(y)
n_correct = (diff==0).sum()
accuracy = n_correct*100.0/n_total

logging.info("Accuracy: {:.2f}% ({} of {}) correct".format(accuracy, n_correct, n_total))


losses = {'discriminator':[], 'gan':[]}
# Create training function
def train_gan(nb_epoch=5000, BATCH_SIZE=10):
	for e in tqdm(range(nb_epoch)):
		image_batch = x_train[np.random.randint(0, len(x_train), size=BATCH_SIZE),:,:,:]
		noise_gen = np.random.uniform(0,1, size=(BATCH_SIZE, 100))
		generated_images = generator.predict(noise_gen)


		# Train the discriminator
		X = np.concatenate((image_batch, generated_images))
		y = np.zeros((2*BATCH_SIZE, 2))
		y[0:BATCH_SIZE,1] = 1
		y[BATCH_SIZE:,0] = 1

		make_trainable(discriminator, True)
		d_loss = discriminator.train_on_batch(X,y)
		losses['discriminator'].append(d_loss) # Add losses to list

		# Train Generator-Discriminator stack on input noise to non-generated output class
		noise_tr = np.random.uniform(0,1,size=(BATCH_SIZE,100))
		y2 = np.zeros((BATCH_SIZE,2))
		y2[:,1] = 1

		make_trainable(discriminator, False)

		g_loss = GAN.train_on_batch(noise_tr, y2)
		losses['gan'].append(g_loss) # Add losses to list


train_gan(nb_epoch=100, BATCH_SIZE=10)
#opt.lr = K.variable(1e-5)
#dopt.lr = K.variable(1e-4)
#train_gan(nb_epoch=2000,BATCH_SIZE=10)

#opt.lr = K.variable(1e-6)
#dopt.lr = K.variable(1e-5)
#train_gan(nb_epoch=2000,BATCH_SIZE=10)

# Save generated images
noise_gen = np.random.uniform(0,1, size=(30, 100))
images = generator.predict(noise_gen)
images*=255
images = images.astype('int')
#import pdb;pdb.set_trace()
print("Saving generated images")
classname = "_".join(selected_class.split())
class_folder = "{}/{}".format(save_dir,classname)
create_collage(images, path='{}/{}.jpg'.format(save_dir, classname))
os.makedirs(class_folder)
for index, img in enumerate(images):
	cv2.imwrite('{}/{}_{}.jpg'.format(class_folder, classname,index), img)



logging.info("Saving model to model folder")
GAN.save("{}/GAN_model.h5".format(save_dir))
discriminator.save("{}/discriminator_model.h5".format(save_dir))
generator.save("{}/generator_model.h5".format(save_dir))

logging.info("Saving summaries as txt")
with open('{}/summaries.txt'.format(save_dir),'w') as sumfile:
	sys.stdout = sumfile
	print("----GAN SUMMARY----")
	GAN.summary()
	print("----discriminator SUMMARY----")
	discriminator.summary()
	print("----generator SUMMARY----")
	generator.summary()
	sys.stdout = sys.__stdout__










