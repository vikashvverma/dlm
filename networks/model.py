import numpy as np
np.random.seed(4160) # Reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def get_2d_cnn(p):
	model = Sequential()
	model.add(Convolution2D(p['nb_filters'][0], 
		p['kernel_size'][0][0], p['kernel_size'][0][1],
		border_mode=p['border_mode'][0],
		input_shape=p['input_shape']
		))

	model.add(Activation(p['activation'][0]))
	model.add(Convolution2D(p['nb_filters'][1], p['kernel_size'][1][0], p['kernel_size'][1][1]))
	model.add(Activation(p['activation'][1]))
	model.add(MaxPooling2D(pool_size=p['pool_size'][0]))
	model.add(Dropout(p['dropout'][0]))

	model.add(Flatten())
	model.add(Dense(p['dense'][0]))
	model.add(Activation(p['activation'][2]))
	model.add(Dropout(p['dropout'][1]))
	model.add(Dense(p['nb_classes']))
	model.add(Activation(p['activation'][3]))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	return model

def get_model(nn_type, parameters=None):
	if parameters == None:
		parameters = {
			'nb_filters':[32, 32],
			'kernel_size':[(3,3), (3,3)],
			'border_mode':['valid', 'valid'],
			'activation':['relu', 'relu', 'relu', 'softmax'],
			'pool_size':[(2,2)],
			'dropout':[.2, .35],
			'dense':[128],
			'input_shape':(150,150, 3),
			'nb_classes':2 # Number of outputs possible, in our case, number of people in dataset
		}

	if nn_type == '2dcnn':
		return get_2d_cnn(parameters)
