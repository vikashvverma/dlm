# Generate images from celebrities from original looking images
from utilities.data_handler import get_imdb_data, split_data, get_data, get_classless_images, equal_shuffle
# from utilities.multi_gpu import make_parallel
from utilities.csv_plot import plot_csv
import argparse
import logging
import math
import numpy as np
import cv2
import pickle
import os
import time
import sys
# from keras.utils import np_utils
from tqdm import tqdm, trange
import numpy as np
# from keras import backend as K
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.models import Model
from collections import OrderedDict
import random
import operator

# Reproducibility 
np.random.seed(2000)
random.seed(2000)

parser = argparse.ArgumentParser(description="Generate images with neural networks")
parser.add_argument("-b", "--batch-size", help="specify batch size")
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-s", "--skipto", help="Skip to the specified index (starting with 0)", type=int)
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

if not args.skipto:
	skip_to_index = -1
else:
	skip_to_index = args.skipto

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

def set_trainable(model, allow_train):
        model.trainable = allow_train
        for l in model.layers:
                l.trainable = allow_train

#x,y = get_data(path='data/imdb-wiki/handpicked_restructured/')
x_train,y_train = get_data( resize=(64,64))
x_test,y_test = get_data( resize=(64,64))
logging.info("Data loaded")
# Because floats improves learning efficiency according to http://datascience.stackexchange.com/questions/13636/neural-network-data-type-conversion-float-from-int
x_train /= 255
x_test /= 255

logging.info("Classes to arrays")
input_shape = x_train.shape[1:]
classes = set(y_train)
nb_classes = len(classes)

logging.info("Categorizes output data")
uniques, ids = np.unique(y_train, return_inverse=True)
cat_y = np_utils.to_categorical(ids, len(uniques))

data_dict_x = OrderedDict()
data_dict_y = OrderedDict()
for nx, ny, ny_vec in zip(x_train,y_train, cat_y):
	data_dict_x[ny] = data_dict_x.get(ny, []) + [nx]
	data_dict_y[ny] = data_dict_y.get(ny, []) + [ny_vec]


# Convert data_dict arrays to numpy
for dict_class in data_dict_x:
	data_dict_x[dict_class] = np.array(data_dict_x[dict_class])
	data_dict_y[dict_class] = np.array(data_dict_y[dict_class])


#(x_train, x_test), (y_train, y_test) = split_data(sx, cat_y, ratio=0.90)

sortedclasses = [classname for (classname, classlist) in sorted(data_dict_x.items(), key=lambda x: len(x[1]), reverse=True)] # Returns a list of classnames sorted by number of samples



for class_index, selected_class in enumerate(sortedclasses):

	if class_index < skip_to_index:
		continue

	class_dir = os.path.join(save_dir, "_".join(selected_class.split()))
	os.makedirs(class_dir)
	x_class = data_dict_x[selected_class]
	y_class = data_dict_y[selected_class]


	opt = Adam(lr=1e-5, decay=1e-14)
	dopt = Adam(lr=1e-4, decay=1e-14)

	# Build the generator model
	g_input = Input(shape=[100])

	H = Dense(64*64*3)(g_input)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)
	H = LeakyReLU(0.2)(H)

	H = Reshape( [64, 64, 3] )(H)
	#H = UpSampling2D(size=(2,2))(H)

	H = Convolution2D(64, 3, 3, border_mode='same')(H)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)
	H = LeakyReLU(0.2)(H)

	H = Convolution2D(128, 3, 3, border_mode='same')(H)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)
	H = LeakyReLU(0.2)(H)

	#H = Convolution2D(256, 3, 3, border_mode='same')(H)
	#H = BatchNormalization(mode=2)(H)
	#H = Activation('relu')(H)
	#H = LeakyReLU(0.2)(H)

	#H = Convolution2D(512, 3, 3, border_mode='same')(H)
	#H = BatchNormalization(mode=2)(H)
	#H = Activation('relu')(H)
	#H = LeakyReLU(0.2)(H)

	H = Deconvolution2D(3, 3, 3, border_mode='same', output_shape=(None, 64,64, 3))(H)
	g_V = Activation('sigmoid')(H)
	generator = Model(g_input, g_V)
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	print("Generator output: {}".format(generator.output_shape))


	d_input = Input(shape=(64,64,3))
	H = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Dropout(0.1)(H)
	H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same')(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(0.1)(H)
	H = Flatten()(H)
	H = Dense(256)(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(0.1)(H)
	d_V = Dense(2,activation='softmax')(H)
	discriminator = Model(d_input, d_V)
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	print("Discriminator output: {}".format(discriminator.output_shape))
	
			
	set_trainable(discriminator, False)

	# Build stacked GAN model
	gan_input = Input(shape=[100])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)
	#GAN = make_parallel(GAN, 2) Multigpu stuff
	GAN.compile(loss='categorical_crossentropy', optimizer=opt)
	
	

	def save_gen(number=16):
		noise = np.random.uniform(0,1,size=[number, 100])
		generated_images = generator.predict(noise)
		
		for index,img in enumerate(generated_images):
			imgpath = "{}/{}_{:06d}.jpg".format(save_path, imgname, index)
			cv2.imwrite(imgpath,img)


	# Pretrain discriminator
	noise = np.random.uniform(0,1, size=[len(x_class), 100])
	generated_images = generator.predict(noise)
	X = np.concatenate((x_class, generated_images))
	n = len(x_class)
	answers = np.zeros(shape=(2*n,2))

	# Set class values (real or generated image)
	answers[:n,1] = 1
	answers[n:,0] = 1

	# Shuffle data
	equal_shuffle(X,answers)
	set_trainable(discriminator, True)
	discriminator.fit(X,answers, nb_epoch=1, batch_size=7)
	predictions = discriminator.predict(X)

	# Measure accuracy of pre-trained discriminator
	predictions_idx = np.argmax(predictions, axis=1)
	answers_idx = np.argmax(answers, axis=1)
	diff = answers_idx-predictions_idx
	n_total = len(answers)
	n_correct = (diff==0).sum()
	accuracy = n_correct*100.0/n_total

	logging.info("Accuracy: {:.2f}% ({} of {}) correct".format(accuracy, n_correct, n_total))

	def save_epoch(cur_epoch, include_model=False, separate_images=False):
		noise_gen = np.random.uniform(0,1, size=(30, 100))
		images = generator.predict(noise_gen)
		images*=255
		images = images.astype('int')

		logging.info("Saving generated images")
		classname = "_".join(selected_class.split())
		create_collage(images, path='{}/{}_{}.jpg'.format(save_dir, cur_epoch, classname))
		if separate_images:
			# Save images in separate folder
			logging.info("Saving images in separate folder")
			for index, img in enumerate(images):
				cv2.imwrite('{}/{}_{}.jpg'.format(class_dir, classname, index), img)
		
		if include_model:
			logging.info("Saving models to model folder")
			GAN.save("{}/{}_{}_GAN_model.h5".format(class_dir, classname, cur_epoch))
			discriminator.save("{}/{}_{}_discriminator_model.h5".format(class_dir, classname, cur_epoch))
			generator.save("{}/{}_{}_generator_model.h5".format(class_dir, classname, cur_epoch))


	losses = {'discriminator':[], 'gan':[]}
	# Create training function
	def train_gan(nb_epoch=5000, batch_size=10, save_frequency=1000, save_model_checkpoints=False):
		datagen = ImageDataGenerator(
			#featurewise_center=True,
			#featurewise_std_normalization=True,
			rotation_range=10,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			horizontal_flip=True
		)
		datagen.fit(x_class) # Returns a generator which returns a tuple (x_batch, y_batch) where y_batch is 
		realimage_generator = datagen.flow(x_class, y_class, batch_size=len(x_class)) # Use this to generate "real" images. (Data augmentation)
		progressbar = trange(nb_epoch, desc="{} ({}/{})".format(selected_class, class_index+1, nb_classes), leave=True)
		for e in progressbar:
			#image_batch = x_class[np.random.randint(0, len(x_class), size=batch_size),:,:,:] 

			real_generated_images = realimage_generator.next()[0]
			image_batch = real_generated_images[np.random.randint(0, len(real_generated_images), size=batch_size),:,:,:]

			noise_gen = np.random.uniform(0,1, size=(batch_size, 100))
			generated_images = generator.predict(noise_gen)
			

			# Train the discriminator
			x = np.concatenate((image_batch, generated_images))
			y = np.zeros((2*batch_size, 2))
			y[0:batch_size,1] = 1
			y[batch_size:,0] = 1

			set_trainable(discriminator, True)
			with K.tf.device('/gpu:1'):
				d_loss = discriminator.train_on_batch(x,y)
			losses['discriminator'].append(d_loss) # Add losses to list

			# Train Generator-Discriminator stack on input noise to non-generated output class
			noise_tr = np.random.uniform(0,1,size=(batch_size,100))
			y2 = np.zeros((batch_size,2))
			y2[:,1] = 1

			set_trainable(discriminator, False) # Discriminator doesnt get trained whilst training the generator
			with K.tf.device('/gpu:1'):
				g_loss = GAN.train_on_batch(noise_tr, y2)
			losses['gan'].append(g_loss) # Add losses to list
		
			if e % save_frequency == 0 and e != 0 and save_frequency != -1:
				save_epoch(e, save_model_checkpoints)

		save_epoch(e, include_model=True)

	if not args.batch_size and not args.batch_size.is_digit():
		bs = 50
		logging.info("Batch size not specified, using default of {}".format(bs))
	else:
		logging.info("Batch size {} was specified in arguments".format(args.batch_size))
		bs = int(args.batch_size)

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

	train_gan(nb_epoch=300000, batch_size=bs, save_frequency=1000)

	#opt.lr = K.variable(1e-6)
	#dopt.lr = K.variable(1e-5)
	#train_gan(nb_epoch=2000,batch_size=bs)

	#opt.lr = K.variable(1e-6)
	#dopt.lr = K.variable(1e-5)
	#train_gan(nb_epoch=2000,batch_size=10)

	# Save generated images


	logging.info("Save losses to file")
	loss_keys = list(losses.keys())
	losslists = [losses[l] for l in loss_keys]
	csvpath = "{}/{}".format(class_dir, "losses.txt")
	with open(csvpath, 'w') as loss_file:
		header_string = "epoch"
		for k in loss_keys:
			header_string += ",%s" % k
		loss_file.write("{}\n".format(header_string))

		for epoch, loss_vals in enumerate(zip(*losslists)):
			line = "{}".format(epoch+1)
			for l in loss_vals:
				line += ",{}".format(l)

			loss_file.write("{}\n".format(line))

	plot_csv(csvpath)
