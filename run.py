from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from utilities.data_handler import get_imdb_data, split_data, get_data
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
	parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
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

	"""
	# Load data via mongodb and such
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


	logging.info("Categorizes output data")
	uniques, ids = np.unique(y, return_inverse=True)
	cat_y = np_utils.to_categorical(ids, len(uniques))
	(x_train, x_test), (y_train, y_test) = split_data(x, cat_y)
	"""



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
		

	def ganbatch_generator(samples_per_epoch=50000, path='data/lfw/lfw_split_cropped/gangen/'):
		# Generator functions that returns [(images, class)]
		classes = os.listdir(path)
		files_by_class = defaultdict(list)
		logging.info("Initializing generator for {} classes from {}".format(len(classes), path))
		for selected_class in classes:
			class_dir = os.path.join(path, selected_class)
			for imgpath in os.listdir(class_dir):
				files_by_class[selected_class].append(os.path.join(class_dir,imgpath))
			
		
		nb_samples = samples_per_epoch / len(files_by_class) # number of samples pr class each epoch
		max_epochs = int(sum([len(files_by_class[k]) for k in files_by_class]) / samples_per_epoch)
		logging.info("Generator initialized, ready to generate next batch...")
		for e in range(max_epochs):
			logging.info("Generator function loading images from epoch {} to memory".format(e))
			image_paths = []
			images = []
			fidx = int(e*nb_samples)
			tidx = int(fidx+nb_samples)
			for selected_class in files_by_class:
				class_vector_idx = np.where(uniques==" ".join(selected_class.split('_')))[0][0]
				class_vector = np.zeros(len(files_by_class))
				class_vector[class_vector_idx] = 1

				image_paths.append((files_by_class[selected_class][fidx:tidx],class_vector))

			for img_path_list,img_class in image_paths:
				for img_path in img_path_list:
					images.append((cv2.imread(img_path), img_class))

			yield np.array(images)

	# Define some variables for easier use
	input_shape = x_train.shape[1:]
	classes = (set(c_train)|set(c_test)|set(gan_classes))
	nb_classes = len(classes)
	logging.info("Building model")

	#TODO: Probably add some shuffle
	
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

	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1,1), border_mode='same'))
	model.add(Activation('relu'))

	model.add(Dropout(0.2))
	
	#model.add(MaxPooling2D(pool_size=(2,2)))

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



			

	gan_generator = ganbatch_generator(samples_per_epoch=50000)
	nb_epoch = 300
	for e in range(nb_epoch):
		logging.info("Epoch {}/{}".format(e+1, nb_epoch))
		genxtrain,genytrain = zip(*next(gan_generator))
		genxtrain = np.array(genxtrain)
		genytrain = np.array(genytrain)
		model.fit(genxtrain, genytrain, batch_size=32, nb_epoch=20, verbose=1, validation_data=(x_test, y_test)) # GAN images

	#model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=50000, nb_epoch=300) # Data augmentation on normal images
	#model.fit(x_train, y_train, batch_size=32, nb_epoch=300,verbose=1, validation_data=(x_test, y_test)) # Normal images, no generation

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





