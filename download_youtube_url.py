#!/usr/bin/python3
from pytube import YouTube
import argparse
import sys
import logging
import os

def get_url_list(path='./vidlist'):
    l = []
    logging.info("Checking if url list {} exists".format(path))
    if (os.path.exists(path)):
        logging.warning("Path: {} did exist".format(path))
    else:
        sys.exit("Path: {} did not exist".format(path))
    
    logging.info("Opening file {}".format(path))
    with open(path, 'r') as url_file:
        for line in url_file:
            url = line.split("#")[0]
            l.append(url)
    return l

def download_url(url, path='../data/videos'):
    logging.info("Checking if folder {} exists".format(path))

    if (os.path.isdir(path)):
        logging.info("Path: {} did exist, skipping creation".format(path))
    elif (os.path.exists(path)):
        sys.exit("ERROR: Path: {} was a file, exiting".format(path))
    else:
        logging.info("Path: {} did not exist, creating directory".format(path))
        os.makedirs(path)
    
    yt = YouTube(url)
    logging.info("Downloading YouTube video '{}'".format(yt.title)) 
    video = yt.get('mp4','720p')
    video.download(path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Download YouTube videos.")
    parser.add_argument("url", help="Url for YouTube video", type=str)
    parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity (Can be specified multiple times for more verbosity)", default=0)
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run, wont modify any files")
    #parser.add_argument("-f", "--file-list", action="store_true", help="")
    args = parser.parse_args()
    
    loglevel = logging.ERROR
    if args.verbosity == 1:
        loglevel = logging.WARNING
    elif args.verbosity == 2:
        loglevel = logging.INFO
    elif args.verbosity == 3:
        loglevel = logging.DEBUG

    logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)
    logging.info("Entered URL {}".format(args.url))
    
    if not args.dry_run:
        download_url(args.url)

