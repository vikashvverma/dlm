from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from utilities.data_handler import get_imdb_data, split_data, get_data, equal_shuffle
from collections import OrderedDict, defaultdict
from tqdm import tqdm, trange
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
	#parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
	parser.add_argument("-s", "--save-data", action="store_true", help="Save data as pickle to model folder")
	parser.add_argument("-i", "--hostname", help="MongoDB Hostname")
	parser.add_argument("-p", "--port", help="MongoDB Port")
	parser.add_argument("-r", "--reload-data", action="store_true", help="Reload data even if pickle file exists")
	parser.add_argument("-n", "--include-ng-classes", action="store_true", help="Include classes that have no generated images")
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

	def show_image(index):
		cv2.imshow(y[index], x[index])
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	# Load LFW data
	logging.info("Loading training data...")
	x_train,c_train = get_data(path='data/lfw/lfw_split_cropped/train/',resize=(64,64))
	logging.info("Loading testing data...")
	x_test,c_test = get_data(path='data/lfw/lfw_split_cropped/test/',resize=(64,64))
	#logging.info("Loading generated data...")
	#x_gen,c_gen = get_data(path='data/lfw/lfw_split_cropped/gangen/',resize=(64,64))


	gan_classes = os.listdir('data/lfw/lfw_split_cropped/gangen/')
	gan_classes = [" ".join(c.split('_')) for c in gan_classes]
	
	if not args.include_ng_classes:
		logging.info("Recreating data lists but only including classes that has generated images.")

		tmp_xtrain,tmp_ctrain = [],[]
		for i in range(len(x_train)):
			if c_train[i] in gan_classes:
				tmp_xtrain.append(x_train[i])
				tmp_ctrain.append(c_train[i])
		x_train = np.array(tmp_xtrain)
		c_train = np.array(tmp_ctrain)

		tmp_xtest,tmp_ctest = [],[]
		for i in range(len(x_test)):
			if c_test[i] in gan_classes:
				tmp_xtest.append(x_test[i])
				tmp_ctest.append(c_test[i])
		x_test = np.array(tmp_xtest)
		c_test = np.array(tmp_ctest)
	
	uniques, ids = np.unique(c_train, return_inverse=True)

	train_uniques, train_ids = np.unique(c_train, return_inverse=True)
	y_train = np_utils.to_categorical(train_ids, len(train_uniques))
	

	test_uniques, test_ids = np.unique(c_test, return_inverse=True)
	y_test = np_utils.to_categorical(test_ids, len(test_uniques))

	#uniques, ids = np.unique(c_gen, return_inverse=True)
	#y_gen = np_utils.to_categorical(ids, len(uniques))
	equal_shuffle(x_train, y_train)

	def get_classname(idx):
		return uniques[idx]

	def get_classvector(classname):
		idx = get_classidx(classname)
		vec = np.zeros(len(uniques))
		vec[idx] = 1
		return vec

	def get_classidx(classname):
		classname = " ".join(classname.split('_'))
		return np.where(classname==uniques)[0][0]
		

	def ganbatch_generator(batch_size=32, mix_real_data=False, path='data/lfw/lfw_split_cropped/gangen/'):
		# Generator functions that returns ([images],[classes])
		classes = os.listdir(path)
		files_by_class = defaultdict(list)
		image_paths = []
		logging.info("Initializing generator for {} classes from {}".format(len(classes), path))
		for selected_class in classes:
			class_dir = os.path.join(path, selected_class)
			for imgpath in os.listdir(class_dir):
				image_paths.append((os.path.join(class_dir,imgpath),selected_class))
		
		np.random.shuffle(image_paths)
		
		batch_idx = 0
		effective_batch_size = batch_size / 2
		batch_x, batch_y = [],[]
		for imgidx, (imgpath, imgclass) in enumerate(image_paths):
			if len(batch_x) % batch_size == 0:
				if imgidx != 0:
					batch_idx += 1
					yield (np.array(batch_x), np.array(batch_y))

				batch_x = []
				batch_y = []
				if mix_real_data:
					batch_x.extend(x_train[int(batch_idx*effective_batch_size):int((batch_idx+1)*effective_batch_size)])
					batch_y.extend(y_train[int(batch_idx*effective_batch_size):int((batch_idx+1)*effective_batch_size)])

			img = cv2.imread(imgpath)
			class_vector_idx = np.where(uniques==" ".join(imgclass.split('_')))[0][0]
			class_vector = np.zeros(len(uniques))
			class_vector[class_vector_idx] = 1

			batch_x.append(img)
			batch_y.append(class_vector)

	# Define some variables for easier use
	input_shape = x_train.shape[1:]
	classes = (set(c_train)|set(c_test)|set(gan_classes))
	nb_classes = len(classes)
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

	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))

	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))

	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.15))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	date_str = time.strftime("%d-%m-%Y_%H-%M-%S")
	#name_str = "{}-acc:{:.2f}-loss:{:.4f}".format(date_str, score[1], score[0])
	name_str = "{}".format(date_str)
	model_folder = "data/models/{}".format(name_str)
	logging.info("Creating model folder {}".format(model_folder))
	os.makedirs(model_folder, exist_ok=True)

	csv_logger_normal = CSVLogger('{}/training_normal.log'.format(model_folder))
	csv_logger_gan = CSVLogger('{}/training_gan.log'.format(model_folder))
	csv_logger_dataaug = CSVLogger('{}/training_dataaug.log'.format(model_folder))


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

	plot_results = {}	

	model.fit(x_train, y_train, batch_size=32, nb_epoch=100,verbose=1, validation_data=(x_test, y_test), callbacks=[csv_logger_normal]) # Normal images, no generation
	model.fit_generator(ganbatch_generator(batch_size=32), samples_per_epoch=int(len(x_train)*2), nb_epoch=100, validation_data=(x_test, y_test), callbacks=[csv_logger_gan]) # GAN aug

	#model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=49984, nb_epoch=300, validation_data=(x_test, y_test), callbacks=[csv_loggerdataaug]) # Data augmentation on normal images
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	#date_str = time.strftime("%d-%m-%Y_%H-%M-%S")
	#name_str = "{}-acc:{:.2f}-loss:{:.4f}".format(date_str, score[1], score[0])
	#model_folder = "data/models/{}".format(name_str)
	#logging.info("Creating model folder {}".format(model_folder))
	#os.makedirs(model_folder, exist_ok=True)

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





