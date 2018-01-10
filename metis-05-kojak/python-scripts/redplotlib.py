"""Plotting functions specific to the redistricting class"""

import os
import glob
import subprocess
from random import shuffle
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from datetime import datetime

from kojak import determine_bounds


"""These dictionaries define various color schemes used when plotting."""
colors = {'green':      '#45735F',
          'light green':'#A6BF80',
          'crimson':    '#DC143C',
          'soft red':   '#BE4248',
          'royal blue': '#4169E1',
          'soft blue':  '#5AA6CF',
          'dark blue':  '#35478C',
          'dark purple':'#36175E',
          'blend purp': '#35478C',
          'white':      '#FFFFFF',
          'gray':       '#999999',
          'soft gray':  '#586473',
          'pale gray':  '#E7DACB',
          'black':      '#000000',
          'light blue': '#56B9D0',
          'egg':        '#FEFEFE',
          'mustard':    '#FBBA42',
          'burnt':      '#F24C27',
          'light black':'#3B3F42',
          'turq':       '#126872',
          'light green':'#18C29C'}

scheme = {'earth':      colors['pale gray'],
          'rep':        colors['soft red'],
          'dem':        colors['dark blue'],
          'uncontested':colors['dark purple'],
          'pt border':  colors['soft gray'],
          'dst border': colors['soft gray']}

lw_pt = 0.1
lw_dstrct = 0.5

palette = [colors['light green'], colors['light black'], colors['turq'],
           colors['burnt'], colors['egg'], colors['mustard'],
           colors['light blue']]


def initialize_plot(precincts, figsize=(10, 10),
                    weights=None, compactness_method=None):
    """
    Initializes a plot with empty precincts.
    ---
    Args:
        precincts: (list) list of precinct objects
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    margin = 0.1
    lon_min, lat_min, lon_max, lat_max = determine_bounds(precincts, margin)
    fc, ec, lw = scheme['earth'], scheme['pt border'], lw_pt
    for pt in precincts:
        patch = PolygonPatch(pt.geometry, fc=fc, ec=ec, linewidth=lw)
        ax.add_patch(patch)


    # lon_min, lat_min = m(lon_min, lat_min)
    # lon_max, lat_max = m(lon_max, lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1)
    ax.axis('off')

    if weights and compactness_method:
        txt = ('Population Weight: {}\n'
               'Compactness Weight: {}\n'
               'Compactness Method: {}\n'
               'Efficiency Gap Weight: {}\n'
               '').format(weights[0], weights[1],
               compactness_method.title(), weights[2])
        # ax.text(0.05, 0.05, txt, fontsize=12,
        #              horizontalalignment='left',
        #              verticalalignment='top',
        #              transform = ax.transAxes)

    for fl in glob.glob('../images/frames/*.png'):
        os.remove(fl)
    save_frame('0_blank')
    return fig, ax


def plot_seeds(ax, clusters):
    """
    Plots the initial seed precincts
    ---
    Args:
        ax:         matplotlib axes object
        clusters:   (list) list of cluster or precinct objects
    """
    ec, lw = scheme['dst border'], lw_dstrct
    for clstr in clusters:
        if clstr.rep_votes > clstr.dem_votes:
            fc = scheme['rep']
        else:
            fc = scheme['dem']
        patch = PolygonPatch(clstr.geometry, fc=fc, ec=ec, linewidth=lw)
        ax.add_patch(patch)

    save_frame('0_seeds')


def highlight_district(ax, clstr, iteration):
    """
    Outlines the cluster provided.
    ---
    Args:
        ax:     matplotlib axes object
        clstr:  (cluster or precinct) a single cluster or precinct object
    """
    fc, ec, lw = clstr.fc, '#000000', 3
    patch = PolygonPatch(clstr.geometry, fc=fc, ec=ec, linewidth=lw)
    ax.add_patch(patch)
    save_frame(iteration)


def update_plot(ax, clstr, iteration):
    """
    Updates the figure with the cluster provided.
    ---
    Args:
        ax:     matplotlib axes object
        clstr:  (cluster or precinct) a single cluster or precinct object
    """
    fc, ec, lw = clstr.fc, scheme['dst border'], lw_dstrct
    patch = PolygonPatch(clstr.geometry, fc=clstr.fc, ec=ec, linewidth=lw)
    ax.add_patch(patch)
    # ttl = 'Iteration: {}'.format(iteration)
    save_frame(iteration)


def final_plot(ax, clusters, iteration=None, color_by_party=False):
    """
    When making a gif, there is no need to update the plot every time.
    Instead, this will plot the final districts.
    ---
    Args:
        ax:     matplotlib axes object
        clusters:   (list) list of cluster objects
    """
    ec, lw =  scheme['dst border'], lw_dstrct
    for clstr in clusters:
        if color_by_party:
            if clstr.rep_votes > clstr.dem_votes:
                fc = scheme['rep']
            else:
                fc = scheme['dem']
        else:
            fc = clstr.fc

        patch = PolygonPatch(clstr.geometry, fc=fc, ec=ec, linewidth=lw)
        ax.add_patch(patch)
    if iteration:
        ttl = 'Iteration: {}'.format(iteration)
        # plt.title(ttl)
    save_frame(iteration)


def label_move_counts(ax, precincts):
    """
    Used for debugging. Labels each precinct with its move count. Best used when
    clustering a subset of the data, as the labels may be illegible when viewing
    an entire state.
    ---
    Args:
        ax:     matplotlib axes object
        clstr:  (precinct) a single precinct object
    """
    for pt in precincts:
        x = pt.centroid[0]
        y = pt.centroid[1]
        ax.text(x, y, str(pt.move_count))


def save_frame(iteration):
    """Saves the current figure"""
    if iteration:
        if not isinstance(iteration, str):
            iteration = str("%05i"%iteration)
    else:
        iteration=''

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig('../images/frames/{}_{}.png'.format(iteration, ts),
                dpi=300, transparent=True)


def make_gif(timestamp, iteration):
    """
    1. Combine all saved frames into a gif image
    2. Print out execution time
    """
    # plt.savefig('../images/frames/last.png', dpi=300, transparent=True)
    save_frame(iteration)
    filename = '../images/gifs/NC_{}.gif'.format(timestamp)
    command = ('convert -delay 5 ../images/frames/*.png '
               '-layers OptimizeFrame {}'
               '').format(filename)
    subprocess.call(command.split())

    # for fl in glob.glob('../images/frames/*.png'):
    #     os.remove(fl)

    filename = '../images/final_plots/NC_{}.png'.format(timestamp)
    plt.savefig(filename, dpi=300)
