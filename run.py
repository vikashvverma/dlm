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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Run neural networks")
	parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
	parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
	parser.add_argument("-s", "--save-data", action="store_true", help="Save data as pickle to model folder")
	parser.add_argument("-i", "--hostname", help="MongoDB Hostname")
	parser.add_argument("-p", "--port", help="MongoDB Port")
	parser.add_argument("-r", "--reload-data", action="store_true", help="Reload data even if pickle file exists")
	args = parser.parse_args()
	loglevel = logging.ERROR
	if args.verbosity == 1:
		loglevel = logging.WARNING
	elif args.verbosity == 2:
		loglevel = logging.INFO
	elif args.verbosity == 3:
		loglevel = logging.DEBUG

	logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)

	if not args.hostname:
		hostname = 'localhost'
		logging.info("No hostname specified, using default of {}".format(hostname))
	else:
		hostname = args.hostname
		logging.info("Hostname specified, using {}".format(hostname))

	if not args.port:
		port = 27017	
		logging.info("No port specified, using default of {}".format(port))
	else:
		port = int(args.port)
		logging.info("Port specified, using {}".format(port))
	pickle_loc = "data/tmp_data.pkl"
	if os.path.exists(pickle_loc) and not args.reload_data:
		logging.info("Loading pickle file")
		with open(pickle_loc, 'rb') as pkl:
			x,y = pickle.load(pkl)
	else:
		if args.reload_data:
			logging.info("--reload-data (-r) was specified")
		logging.info("Executing data_handler")
		#x,y = get_data(path='data/cropped/lfw',resize=(64,64), min_examples=5)
		x,y = get_imdb_data(collection="56f492c9fba69dbd2439b7975e9e279e_cropped", people_limit=10, port=port, hostname=hostname)
		logging.info("Saving data as pkl")
		with open(pickle_loc, 'wb') as pkl:
			pickle.dump((x,y), pkl)
	
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
	classes = set(y)
	nb_classes = len(classes)

	def show_image(index):
		cv2.imshow(y[index], x[index])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	logging.info("Categorizes output data")
	uniques, ids = np.unique(y, return_inverse=True)
	cat_y = np_utils.to_categorical(ids, len(uniques))
	(x_train, x_test), (y_train, y_test) = split_data(x, cat_y)
	
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


	datagen = ImageDataGenerator(
		featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range=20,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
		channel_shift_range=0.1
	)
	datagen.fit(x_train)

	model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=50000, nb_epoch=300)
	#model.fit(x_train, y_train, batch_size=50, nb_epoch=1000,verbose=1, validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)

	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	if not args.dry_run:
		date_str = time.strftime("%d-%m-%Y_%H-%M-%S")
		name_str = "{}-acc:{:.2f}-loss:{:.4f}".format(date_str, score[1], score[0])
		model_folder = "data/models/{}".format(name_str)
		logging.info("Creating model folder {}".format(model_folder))
		os.makedirs(model_folder, exist_ok=True)

		logging.info("Saving model as {}/model.h5".format(model_folder))
		model.save("{}/model.h5".format(model_folder))

		logging.info("Saving summary as txt")
		with open('{}/summary.txt'.format(model_folder),'w') as sumfile:
			sys.stdout = sumfile
			print("x_train.shape: {}".format(x_train.shape))
			print("y_train.shape: {}".format(y_train.shape))
			model.summary()
			sys.stdout = sys.__stdout__
		if args.save_data:
			logging.info("Saving data to pickle file because --save-data given")
			pkl_data = {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test, 'uniques':uniques, 'ids':ids}
			with open('{}/data.pkl'.format(model_folder), 'wb') as pkl:
				pickle.dump(pkl_data, pkl)





