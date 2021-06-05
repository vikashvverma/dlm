#!/usr/bin/python3
# This file will attempt to detect and crop out faces from images.
from pymongo import MongoClient
import argparse
import cv2
import hashlib
import logging
import os
import sys
import shutil
import multiprocessing

parser = argparse.ArgumentParser(description="Use mongodb collection to look up images and crop aswell as add a new collection for it.")
parser.add_argument("collection", help="MongoDb collection", type=str)
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-c", "--new-collection", help="Name of new collection, default is 'collection-cropped'")
parser.add_argument("-f", "--force", action="store_true", help="Force, overwrite any files with conflicts")
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

def get_db():
	if not args.hostname:
		hostname = 'localhost'
		logging.debug("No hostname specified, using default of {}".format(hostname))
	else:
		hostname = args.hostname
		logging.debug("Hostname specified, using {}".format(hostname))

	if not args.port:
		port = 27017	
		logging.debug("No port specified, using default of {}".format(port))
	else:
		port = int(args.port)
		logging.debug("Port specified, using {}".format(port))

	mc = MongoClient(hostname, port)
	db = mc['dlm']
	return db

def get_collections():
	db = get_db()
	new_collection = None
	if not args.new_collection:
		new_collection = db[args.collection+"_cropped"]
	else:
		new_collection = db[args.new_collection]

	collection = db[args.collection]

	return collection,new_collection



def face_crop(record):
	pre_path = "data/cropped/{}".format(args.collection)

	image = cv2.imread(record['full_path'])
	face_cascade = cv2.CascadeClassifier('utilities/resources/haarcascade/haarcascade_frontalface_alt2.xml')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(40, 40),
		flags=0
	)

	if len(faces) != 1: # Only images with 1 face valid in training
		return None

	fp = record['full_path']
	subdir = os.path.split(os.path.dirname(fp))[-1]
	fn,ext = os.path.splitext(os.path.basename(fp))
	face_records = []
	for i,(x,y,w,h) in enumerate(faces):
		image_path = pre_path+'/{sub}_{fn}_{index}{ext}'.format(sub=subdir, fn=fn, index=i, ext=ext)

		face_record = record

		# Cropping image to size
		cropped = image[y:y+h, x:x+w]
		cropped = cv2.resize(cropped, (64,64))
		# Saving cropped image
		logging.debug("Saving {}".format(image_path))
		cv2.imwrite(image_path, cropped)
		

		face_record['old_path'] = record['full_path']
		face_record['full_path'] = image_path
		face_record['old_id'] = face_record.pop('_id',-1)

		face_records.append(face_record)

	return face_records


if __name__ == '__main__':
	pre_path = "data/cropped/{}".format(args.collection)

	collection, new_collection = get_collections()

	if collection.find({}).count() == 0:
		logging.error("Collection was empty")
		sys.exit(1)
	
	if new_collection.count() != 0 and args.force:
		logging.warn("--force specified, dropping collection")
		new_collection.drop()

	def show(img,name="image"):
		cv2.imshow(name, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	def showc(img, croploc, name="image"):
		x,y,w,h = croploc
		crop = img[y:y+h, x:x+w]
		show(crop, name)

	def sa(img, croplocs, name="image"):
		for c in croplocs:
			showc(img, c, name)


	cropped_path = "data/cropped/{}".format(args.collection)

	if not os.path.isdir(cropped_path) and not args.force:
		os.makedirs(cropped_path)
	elif os.path.isdir(cropped_path) and args.force:
		logging.warn("--force specified, deleting image folder")
		shutil.rmtree(cropped_path)
		os.makedirs(cropped_path)
	elif not os.path.isdir(cropped_path) and args.force:
		logging.info("--force specified, but no folder found. Continuing as usual")
		os.makedirs(cropped_path)
	else:
		logging.error("Folder already exists, specify --force if you want overwrite")
		sys.exit(1)
	
	cursor = collection.find()

	loaded_cursor = [r for r in cursor]

	pool = multiprocessing.Pool()
	raw_records = pool.map(face_crop, loaded_cursor)
	records = [r for r in raw_records if r] # Remove empty lists / images without faces

	logging.debug("Begining mongoDb inserts")
	for i,r in enumerate(records):
		new_collection.insert_many(r)
		
	
	cursor.close()
