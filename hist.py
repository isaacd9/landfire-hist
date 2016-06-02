#!/usr/bin/python

import sys
import os
import os.path as path
import argparse
import csv

import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

unburnables = set([91,93,98,99])

cnames = {
        1: "Short Grass",
        2: "Timber Grass and Understory",
        3: "Tall Grass",
        4: "Chaparral",
        5: "Brush",
        6: "Dormant Brush",
        7: "Southern Rough",
        8: "Compact Timber Litter",
        9: "Hardwood Litter",
        10: "Timber Understory",
        11: "Light Slash",
        12: "Medium Slash",
        13: "Heavy Slash",
        91: "Urban",
        93: "Snow/Ice",
        98: "Water",
        99: "Barren/No Data"
}

ccolors = {
        1: "#00FF00",
        2: "#00CB02",
        3: "#2C8E02",
        4: "#C26701",
        5: "#9B6702",
        6: "#642D00",
        7: "#690004",
        8: "#9D009C",
        9: "#9900CE",
        10: "#9402EE",
        11: "#33330D",
        12: "#323334",
        13: "#343561",
        91: "#FCFB0F",
        93: "#9AFFFE",
        98: "#0001FB",
        99: "#000000"
}

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


def normalize(hists):
    d = None
    k = 0

    for h in hists:
        if len(h) > k:
            d = h

    ret = []
    for h in hists:
        for k in d:
            if k not in h:
                h[k] = 0

        sum_h = sum(h.values())
        normal = map(lambda (x,y): (x, y / float(sum_h)), h.items())
        ret.append(dict(normal))

    return ret

def write_csv(hists, filenames, sort_by_values=False, save=False):
    """ Writes a CSV value of the fuels followed by the amounts
    """
    writer = csv.writer(open(filenames[0].split('.')[0] + '.csv', 'wb'))

    field = int(sort_by_values)
    keys = [key for (key, value) in sorted(hists[0].items(), key=lambda kv: kv[field], reverse=sort_by_values)]
    writer.writerow(keys)
    for hist in hists:
        writer.writerow([value for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)])

def plot_line_chart(hists, filenames, sort_by_values=False, save=False, unburnables_on=True):
    """
    Builds a line chart of the histograms"""

    series = dict()

    for h in hists:
        for key, val in h.items():
            if key not in series:
                series[key] = []

            series[key].append(val)

    xticks = range(len(filenames))
    xticks[0] += .05
    xticks[-1] -= .05

    for key, value in sorted(series.items(), key=lambda t: t[0]):
        if unburnables_on or (key not in unburnables):
            l = cnames[key] if key in cnames else key
            plt.plot(xticks, value, marker='o', linestyle='--', color=ccolors[key], label=l, ms=7)
            plt.annotate(s=l, xy=(.025, .005 + value[0]))

    normalized_filenames = map(lambda f: f.split('/')[-1], filenames)
    fontP = FontProperties()
    fontP.set_size('small')

    plt.legend(prop = fontP).draggable()

    plt.xticks(xticks, normalized_filenames)
    plt.title(normalized_filenames[0].split('-')[0] + ' fuels')

    plt.show()


def plot_hists_vert(hists, filenames, sort_by_values=False, save=False):
    """
    Builds a plot of the image"""

    field = int(sort_by_values)

    keys = [key for (key, value) in sorted(hists[0].items(), key=lambda kv: kv[field], reverse=sort_by_values)]

    binned_data_sets = list()
    for hist in hists:
        binned_data_sets.append([value for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)])

    hist_range = len(hists)
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))


    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(0, 10, len(keys) + 1)
    centers = .5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)
    centers[0] = 0

    # Cycle through and plot each histogram
    ax = plt.subplot(111)
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - map(lambda l: l/2, binned_data)
        ax.barh(centers, binned_data, height=heights, left=lefts)

    ax.set_xticks(x_locations)
    ax.set_xticklabels(filenames)

    ax.set_ylabel("Data values")
    ax.set_yticklabels(keys)
    ax.set_yticks(bin_edges)

    plt.show()


def plot_hist(hist, filename, sort_by_values=False, save=False):
    """
    Builds a plot of the image"""

    field = int(sort_by_values)

    ordered_keys = [key for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)]
    ordered_values = [value for (key, value) in sorted(hist.items(), key=lambda kv: kv[field], reverse=sort_by_values)]

    plt.xlabel('Fuels')
    plt.ylabel('Percent')

    plt.title(filename + ' fuels')

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
        out_path = path.join(os.getcwd(), path.basename(filename) + '.plot.png')
        print "Writing file to " + out_path

        plt.savefig(out_path)

    else:
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Build histogram")

    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',)
    parser.add_argument('--sorted', dest='sorted_', action='store_true')
    parser.add_argument('-o', action='store_true', dest='output')
    parser.add_argument('-U', action='store_true', dest='unburnables')

    argv = parser.parse_args()

    hists = []
    filenames = []
    for filename in argv.filenames:
        hist = build_hist(filename)
        hists.append(hist)
        filenames.append(filename)

    hists = normalize(hists)
    plot_line_chart(hists, filenames, argv.sorted_, argv.output, not argv.unburnables)
    #write_csv(hists, filenames, argv.sorted_, argv.output)

