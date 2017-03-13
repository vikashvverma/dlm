import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os
import sys


def plot_csv(csvpath, plotoutput=None):
	if not os.path.isfile(csvpath):
		sys.exit("ERROR, file not found")

	file_directory = os.path.dirname(csvpath)
	filename = os.path.basename(csvpath)
	fn,ext = os.path.splitext(filename)

	csv_rows = []
	with open(csvpath, 'r') as csvfile:
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

	if not plotoutput:
		plt.savefig("{}/{}.png".format(file_directory, fn))
	else:
		plt.savefig(plotoutput)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Plots .csv files with headers")
	parser.add_argument("path", help="CSV file path", type=str)
	parser.add_argument("-o", "--output", help="Output filepath")
	args = parser.parse_args()
	if not args.output:
		op = None
	else:
		op = args.output
		
	plot_csv(args.path, op)
