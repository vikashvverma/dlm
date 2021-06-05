import argparse
import cv2
import logging
import os
import sys
import multiprocessing


parser = argparse.ArgumentParser(description="Parse LFW dataset and crop faces")
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-f", "--force", action="store_true", help="Force and overwrite files that exists")

args = parser.parse_args()

loglevel = logging.ERROR

if args.verbosity == 1:
	loglevel = logging.WARNING
elif args.verbosity == 2:
	loglevel = logging.INFO
elif args.verbosity == 3:
		loglevel = logging.DEBUG
	
logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)

def lfw_face_crop(image_path):
	
	filename = os.path.basename(image_path)
	dirname = os.path.basename(os.path.dirname(image_path))

	cropped_dir = "data/cropped/lfw/{}".format(dirname)
	cropped_path = "{}/{}".format(cropped_dir, filename)
	# Skip if exists
	if os.path.exists(cropped_path):
		if args.force:
			logging.warn("--force is set, removing file")
			os.remove(cropped_path)
		else:
			return False

	logging.debug("Creating folder for {}".format(filename))
	os.makedirs(cropped_dir, exist_ok=True)

	image = cv2.imread(image_path)
	face_cascade = cv2.CascadeClassifier('utilities/resources/haarcascade/haarcascade_frontalface_alt2.xml')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(40, 40),
		flags=0
	)

	if len(faces) != 1:
		logging.info("Found {} faces in {}".format(len(faces), image_path))
		return False

	# Why for-loop for only one image? MEH
	for i,(x,y,w,h) in enumerate(faces):
		cropped = image[y:y+h, x:x+w]
		cropped = cv2.resize(cropped, (64,64))
		logging.debug("Saving {}".format(cropped_path))
		cv2.imwrite(cropped_path, cropped)

	return True


if __name__ == '__main__':
	logging.info("Getting all image paths")
	files = []
	for dirpath,dirnames,filenames in os.walk('data/lfw/lfw'):
		for f in filenames:
			files.append('{}/{}'.format(dirpath,f))
	pool = multiprocessing.Pool()
	success = pool.map(lfw_face_crop,files)















