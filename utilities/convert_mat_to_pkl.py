#!/usr/bin/python3
import pickle
import scipy.io
import argparse
import sys
import logging
import os

parser = argparse.ArgumentParser(description="Parse matlab (.mat) files and export them as pickle files with the same name (.pkl)")
parser.add_argument("path", help="matlab file path", type=str)
parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
parser.add_argument("-f", "--force", action="store_true", help="Force, overwrite any files with conflicts")
parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
args = parser.parse_args()

def save_object(obj, filename):
	with open(filename, 'wb') as outfile:
		pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)

def convert(path):
	logging.info("Loading matfile to memory")
	matfile = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
	dir_path = os.path.dirname(path)
	filename,extension= os.path.splitext(os.path.basename(path))
	newfile = os.path.join(dir_path, filename+'.pkl')
	
	if extension != ".mat":
		sys.exit("File extension not .mat, quitting...")

	if not args.dry_run:
		logging.info("Saving picklefile: {}".format(newfile))
		if os.path.exists(newfile):
			if args.force:
				logging.info("File exists, but force argument specified. Overwriting...")
				save_object(matfile, newfile)
			else:
				sys.exit("File exists already, exiting... (specifiy --force if you want to overwrite)")
		else:
			save_object(matfile, newfile)
	else:
		logging.info("--dry-run specified, quitting...")
		sys.exit(0)

if __name__ == '__main__':
    loglevel = logging.ERROR
    if args.verbosity == 1:
        loglevel = logging.WARNING
    elif args.verbosity == 2:
        loglevel = logging.INFO
    elif args.verbosity == 3:
        loglevel = logging.DEBUG

    logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)
    logging.info("Entered path {}".format(args.path))

    convert(args.path)
