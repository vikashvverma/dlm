from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from networks.model import get_model
from utilities.data_handler import get_imdb_data
import argparse
import logging
import numpy as np

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Load data for use in artificial neural networks")
	parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
	parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
	parser.add_argument("-i", "--hostname", help="MongoDB Hostname")
	parser.add_argument("-p", "--port", help="MongoDB Password")
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

	
	x,y = get_imdb_data(collection="56f492c9fba69dbd2439b7975e9e279e_cropped", people_limit=100)
	logging.info("Data loaded")
	logging.info("Getting numpy RNG state")
	logging.info("Shuffling X")
	rng_state = np.random.get_state()
	np.random.shuffle(x)
	logging.info("Setting numpy RNG state")
	np.random.set_state(rng_state)
	logging.info("Shuffling Y")
	np.random.shuffle(y)

	input_shape = x[0].shape
	nb_classes = len(set(y))


	parameters = {
			'nb_filters':[32, 32],
			'kernel_size':[(3,3), (3,3)],
			'border_mode':['valid', 'valid'],
			'activation':['relu', 'relu', 'relu', 'softmax'],
			'pool_size':[(2,2)],
			'dropout':[.2, .35],
			'dense':[128],
			'input_shape':input_shape,
			'nb_classes':nb_classes
		}


