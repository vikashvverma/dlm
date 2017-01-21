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

parser = argparse.ArgumentParser(description="Use mongodb collection to look up images and crop aswell as add a new collection for it.")
parser.add_argument("collection", help="MongoDb collection", type=str)
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-c", "--new-collection", help="Name of new collection, default is 'collection-cropped'")
#parser.add_argument("-f", "--force", action="store_true", help="Force, overwrite any files with conflicts")
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

	pre_path = "data/cropped/{}".format(args.collection)


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
	db = mc['dlm']

	collection = db[args.collection]

	new_collection = None
	if not args.new_collection:
		new_collection = db[args.collection+"_cropped"]
	else:
		new_collection = db[args.new_collection]

	if collection.find({}).count() == 0:
		logging.error("Collection was empty")
		sys.exit(1)

	def face_crop(record):
		
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

		if len(faces) == 0: # No faces detected
			return []

		fp = record['full_path']
		subdir = os.path.split(os.path.dirname(fp))[-1]
		fn,ext = os.path.splitext(os.path.basename(fp))

		for i,(x,y,w,h) in enumerate(faces):
			image_path = pre_path+'/{sub}_{fn}_{index}{ext}'.format(sub=subdir, fn=fn, index=i, ext=ext)

			face_record = record

			# Cropping image to size
			cropped = image[y:y+h, x:x+w]
			cropped = cv2.resize(cropped, (150,150))
			# Saving cropped image
			logging.debug("Saving {}".format(image_path))
			cv2.imwrite(image_path, cropped)
			

			face_record['old_path'] = record['full_path']
			face_record['full_path'] = image_path
			face_record['old_id'] = face_record.pop('_id',-1)

			# Insert the record to mongodb
			oid = new_collection.insert_one(face_record)
			logging.debug("Inserted record with id {}".format(oid.inserted_id))


	cursor = collection.find(no_cursor_timeout=True)
	for r in cursor:
		if new_collection.find({'old_id':r['_id']}).limit(1).count() != 0:
			logging.debug("{} found in DB, skipping...".format(r['full_path']))
			continue
		face_crop(r)
		
	cursor.close()