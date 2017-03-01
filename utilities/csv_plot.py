import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Plots .csv files with headers")
parser.add_argument("path", help="CSV file path", type=str)
parser.add_argument("-o", "--output", help="Output filepath")
args = parser.parse_args()

if not os.path.isfile(args.path):
	sys.exit("ERROR, file not found")

file_directory = os.path.dirname(args.path)
filename = os.path.basename(args.path)
fn,ext = os.path.splitext(filename)

csv_rows = []
with open(args.path, 'r') as csvfile:
	plots = csv.reader(csvfile, delimiter=',')
	for r in plots:
		csv_rows.append(r)
		
headers = csv_rows[0]
data_rows = csv_rows[1:]

data_dict = {}


for row in data_rows:
	for index, val in enumerate(row):
		data_dict.setdefault(headers[index], []).append(val)

x = data_dict[headers[0]]

for k,v in data_dict.items():
	if k != headers[0]:
		#print('{}: {}'.format(k,v))
		plt.plot(x,v, label=k)
plt.legend()

if not args.output:
	plt.savefig("{}/{}.png".format(file_directory, fn))
else:
	plt.savefig(args.output)
