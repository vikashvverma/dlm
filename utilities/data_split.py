from dlm.utilities.data_handler import split_data, get_data, equal_shuffle
import argparse
import shutil
import cv2
import logging
import os
import sys

parser = argparse.ArgumentParser(description="Split a dataset structured like LFW to test and train dataset")
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-f", "--force", action="store_true", help="Force and overwrite files that exists")
parser.add_argument("path", help="Path to original dataset")
parser.add_argument("output_path", help="Path to where the splitted dataset should be output")
parser.add_argument("-r", "--ratio", help="Ratio of the test and train dataset")

args = parser.parse_args()

loglevel = logging.ERROR

if args.verbosity == 1:
	loglevel = logging.WARNING
elif args.verbosity == 2:
	loglevel = logging.INFO
elif args.verbosity == 3:
		loglevel = logging.DEBUG
	
logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)

outpath = args.output_path
inpath = args.path
if os.path.exists(outpath) and not args.force:
	logging.error("Output path exists")
	sys.exit("Output path exists, quitting...")
elif os.path.exists(outpath) and args.force:
	logging.warning("Output path exists, and force specified, deleting...")
	shutil.rmtree(outpath)
	
logging.info("Creating output folders...")
os.makedirs(outpath)
os.makedirs(os.path.join(outpath,'train'))
os.makedirs(os.path.join(outpath,'test'))
logging.info("Loading data to memory")
x,y = get_data(path=inpath, min_examples=5)
data_dict = {}

# Loop through classes
	# Split each class
	# Save in both train and test

for xe, ye in zip(x,y):
	data_dict[ye] = data_dict.get(ye, []) + [xe]

for person in data_dict:
	logging.info("Creating folders for {}".format(person))
	classdir_train = os.path.join(outpath, "train", "{}".format("_".join(person.split())))
	classdir_test = os.path.join(outpath, "test", "{}".format("_".join(person.split())))
	os.makedirs(classdir_train)
	os.makedirs(classdir_test)

	train,test = split_data(data_dict[person], ratio=0.8)[0]
	img_idx = 1

	for img in train:
		img_path = os.path.join(classdir_train, "{}_{:04d}.jpg".format("_".join(person.split()), img_idx))
		cv2.imwrite(img_path, img)
		img_idx += 1

	for img in test:
		img_path = os.path.join(classdir_test, "{}_{:04d}.jpg".format("_".join(person.split()), img_idx))
		cv2.imwrite(img_path, img)
		img_idx += 1

