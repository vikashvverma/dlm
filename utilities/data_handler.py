from pymongo import MongoClient
import argparse
import cv2
import logging
import numpy as np
import os
import sys
import time

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Load data for use in artificial neural networks")
	parser.add_argument("-c","--collection", help="MongoDb collection")
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

def get_imdb_data(collection, hostname='localhost', port=27017, people_limit=-1):
	logging.info("Beginning process of loading {} to memory".format(collection))
	mc = MongoClient(hostname, port)
	db = mc['dlm']
	collection = db[collection]
	
	x_data = []
	y_data = []

	if people_limit > 0:
		logging.info("People limit is set to {}".format(people_limit))
		# Get people in order of number of entries
		people = {}
		cursor = collection.find()
		for r in cursor:
			people[r['name']] = people.get(r['name'], 0) + 1
		sorted_people = [(k, people[k]) for k in sorted(people, key=people.get, reverse=True)]
		selected_people = []
		for i,(k,v) in enumerate(sorted_people):
			if i >= people_limit:
				break
			selected_people.append((k,v))

		logging.info("Loading {} pictures into memory".format(sum([v for k,v in selected_people])))
		for people_index,(k,_) in enumerate(selected_people):
			cur = collection.find({'name':k})

			logging.debug("Loading pictures of {} ({}/{})".format(k, people_index+1, len(selected_people)))
			for r in cur:
				x_data.append(cv2.imread(r['full_path']))
				y_data.append(r['name'])

	else: # If people_limit is not specified, get all
		logging.info("No people limit is set, expect lots of memory consumed")
		cursor = collection.find()
		logging.info("Loading {} pictures into memory".format(cursor.count()))
		for r in cursor:
			x_data.append(cv2.imread(r['full_path']))
			y_data.append(r['name'])

	return np.array(x_data, dtype=np.uint8), np.array(y_data, dtype=np.str)


def get_lfw_data(path='data/lfw/lfw/', people_limit=-1):
	x_data,y_data = [],[]
	for i,(subdir, dirs, files) in enumerate(os.walk(path)):
		person_name = ' '.join(os.path.basename(subdir).split('_'))
		if i >= people_limit:
			break
		for f in files:
			x_data.append(cv2.imread('{}/{}'.format(subdir, f)))
			y_data.append(person_name)
			print(subdir,f)
	return np.array(x_data, dtype=np.uint8), np.array(y_data, dtype=np.str)

def split_data(*data, ratio=0.8):
	l = []
	for dset in data:
		m = int(len(dset)*ratio)
		l.append((dset[:m], dset[m:]))
	return l

if __name__ == '__main__':
	S = time.time()
	#x,y = get_imdb(collection="56f492c9fba69dbd2439b7975e9e279e_cropped", people_limit=100)
	x,y = get_lfw_data(people_limit=10)
	import pdb;pdb.set_trace()
	print("Loaded to memory after {:.2f} seconds".format(time.time()-S))
