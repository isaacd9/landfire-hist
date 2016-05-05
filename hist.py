#!/usr/bin/python

import sys
import gdal
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(hist, filename, sort_by_values=False, save=False):
    """
    Builds a plot of the image"""

    field = int(sort_by_values)

    ordered_keys = [key for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)]
    ordered_values = [value for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)]

    plt.xlabel('Fuels')
    plt.ylabel('Percent')
    plt.title(filename + ' fuels by amount')
    plt.xticks(
            np.arange(
            len(list(hist))) + .10,
            ordered_keys
            )

    plt.bar(
            range(len(hist)),
            ordered_values,
            align='center'
    )

    #plt.set_ticks(np.arange(0, len(list(hist)), 1))

    if save:
        plt.savefig(filename + '.hist.png')

    else:
        plt.show()


def build_hist(filename):
    """
    Returns a dictionary of fuel types to values"""

    try:
        img = gdal.Open(filename)
        "opened " + filename
    except(e):
        print 'Unable to open %s' % filename
        print e

    try:
        band = img.GetRasterBand(1)
    except(e):
        print 'Raster band not found'
        print e

    metadata = band.GetMetadata()
    model_values = map(int, metadata['FUEL_MODEL_VALUES'].split(','))


    original_hist = band.GetHistogram(0,100,100)
    transformed_hist = {}

    for bucket in model_values:
        if bucket is not 0:
            transformed_hist[bucket] = original_hist[bucket]

    return transformed_hist


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Build histogram")

    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',)
    parser.add_argument('--sorted', dest='sorted_', action='store_true')
    parser.add_argument('-o', action='store_true', dest='output')

    argv = parser.parse_args()

    for filename in argv.filenames:
        hist = build_hist(filename)
        plot_hist(hist, filename, argv.sorted_, argv.output)

