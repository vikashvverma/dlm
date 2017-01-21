#!/usr/bin/python3
# This file will parse .mat files from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
from pymongo import MongoClient
import argparse
import sys
import logging
import os
import scipy.io as sio
import numpy as np
import hashlib as hl

parser = argparse.ArgumentParser(description="Parse matlab (.mat) files and add them to a local mongoDb")
parser.add_argument("path", help="matlab file path", type=str)
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-f", "--force", action="store_true", help="Force, overwrite any files with conflicts")
parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
parser.add_argument("-i", "--hostname", help="MongoDB Hostname")
parser.add_argument("-p", "--port", help="MongoDB Password")
args = parser.parse_args()

if __name__ == '__main__':
	loglevel = logging.ERROR
	if args.verbosity == 1:
		loglevel = logging.WARNING
	elif args.verbosity == 2:
		loglevel = logging.INFO
	elif args.verbosity == 3:
		loglevel = logging.DEBUG

	logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)
	path = args.path
	path = "D:/dev/deep-imdb/dlm/data/imdb-wiki/ex/imdb_crop/imdb.mat" # TODO: remove

	dirs = []
	newpath = None
	temp_path = path
	while os.path.split(temp_path)[1] != '':
		dirs.insert(0, os.path.split(temp_path)[1])
		temp_path = os.path.split(temp_path)[0]
	if dirs[dirs.index('data')-1] == 'dlm':
		logging.info("data folder was found in dlm folder, assuming you want relative paths")
		newpath = os.path.join(*dirs[dirs.index('data'):-1]).replace('\\','/')
	else:
		logging.info("Didn't understand path, using full path for db")
		newpath = path


	logging.info("Using file {}".format(path))
	
	filename,extension= os.path.splitext(os.path.basename(path))
	if extension != ".mat":
		sys.exit("File extension not .mat, quitting...")

	md5hash = hl.md5(open(path,'rb').read()).hexdigest()
	logging.info("MD5 of file: {}".format(md5hash))
	
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

	mc = MongoClient(hostname, port)
	logging.info("Loading .mat file to memory")
	matfile = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
	logging.info(".mat file loaded to memory")

	db = mc['dlm']
	collection = db[md5hash]

	if collection.count() != 0:
		logging.info("Collection {} not empty".format(md5hash))
		if args.force:
			logging.info("Force parameter set, dropping collection and repopulating")
			collection.drop()
		else:
			sys.exit("Collection already exists (use --force or -f ] to overwrite)")
	else:
		logging.info("Collection {} empty. continuing...".format(md5hash))
	imdb = matfile['imdb']

	posts = []
	
	for i in range(len(getattr(imdb, 'full_path'))):
		post = {'_id':i}
		for field in [x for x in imdb._fieldnames if x != 'celeb_names']: # Loop through every field name except celeb_names
			
			# Prepend folders to path, just for clarification in db
			fval = getattr(imdb, field)[i]
			if field == 'full_path':
				fval = os.path.join(newpath, fval).replace('\\','/')

			if isinstance(fval, np.ndarray):
				tmp_list = list(fval)
				for lindex,litem in enumerate(tmp_list):
					tmp_list[lindex] = litem.item()
				post[field] = tmp_list

			elif isinstance(fval, np.generic):
				post[field] = fval.item()
			else:
				post[field] = fval
		posts.append(post)

	if not args.dry_run:
		logging.info("Inserting documents in database")			
		collection.insert_many(posts)	
		logging.info("{} documents inserted".format(len(posts)))
	else:
		logging.info("--dry_run is set, no action will be performed on database")

