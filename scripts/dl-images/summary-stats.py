#!/usr/bin/python
import argparse
from numpy import genfromtxt
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prefix')
args = parser.parse_args()
for architectureid in range(1, 11):
	my_data = genfromtxt('dl-{}-summarized-{}.csv'.format(args.prefix,architectureid), delimiter=',')
	print("%s,%.4f,%.4f,%.4f,%.4f" % (architectureid,my_data.min(), my_data.max(), my_data.mean(), my_data.std()))
