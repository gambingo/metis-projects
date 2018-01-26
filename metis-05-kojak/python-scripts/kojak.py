"""
Metis Project Kojak: Final Project
Metis Data Science Bootcamp
J. Gambino
December 2017
---
Various helper functions
"""

import pickle
from datetime import datetime
from shapely.geometry import shape
import numpy as np


def save_pickle(obj, filename):
    """
    pickles a python object
    ---
    Args:
        obj:        Any python object
        filename:   (string) filename, excluding path and extension
    """
    filepath = '../images/pickled_objects/'
    #ts = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    with open(filepath + filename + '.p', 'wb') as picklefile:
        pickle.dump(obj, picklefile)


def load_pickle(filename):
    """
    loads a pickled object
    ---
    Args:
        filename (string):  filname of pickle file stored in the filepath
                            referenced below.
    """
    #filepath = '../images/pickled_objects/'
    filepath = '../images/pulled_from_AWS/'
    with open(filepath + filename + '.p', 'rb') as picklefile:
        obj = pickle.load(picklefile)
    return obj


def determine_bounds(clusters, margin):
    """
    Determines global bounds for all clusters.
    This is used when initializing a plot.
    ---
    Args:
        clusters:   a list or array of cluster.cluster, precinct.precinct
                    objects, or the raw dictionary data files.
    """
    if isinstance(clusters[0], dict):
        # Raw Data
        geometry = shape(clusters[0]['geometry'])
        lon_min, lat_min, lon_max, lat_max = geometry.bounds

        for clstr in clusters[1:]:
            geometry = shape(clstr['geometry'])
            lon_min_i, lat_min_i, lon_max_i, lat_max_i = geometry.bounds
            lon_min = lon_min_i if lon_min > lon_min_i else lon_min
            lat_min = lat_min_i if lat_min > lat_min_i else lat_min
            lon_max = lon_max_i if lon_max < lon_max_i else lon_max
            lat_max = lat_max_i if lat_max < lat_max_i else lat_max
    else:
        # Cluster (or Precinct) objects
        lon_min, lat_min, lon_max, lat_max = clusters[0].geometry.bounds

        for clstr in clusters[1:]:
            lon_min_i, lat_min_i, lon_max_i, lat_max_i = clstr.geometry.bounds
            lon_min = lon_min_i if lon_min > lon_min_i else lon_min
            lat_min = lat_min_i if lat_min > lat_min_i else lat_min
            lon_max = lon_max_i if lon_max < lon_max_i else lon_max
            lat_max = lat_max_i if lat_max < lat_max_i else lat_max

    lon_min -= margin
    lat_min -= margin
    lon_max += margin
    lat_max += margin

    return lon_min, lat_min, lon_max, lat_max


def slice_up_a_state(data, lat_north=None, lat_south=None,
                           long_east=None, long_west=None):
    """
    Returns a subset of the state data that is either of north or south of the
    specified latitude and/or east or west of the specified longitute.
    Args:
        lat:    (float) Latitude to split on. If None, won't split north/south
        lat_direction:  (string) Split north or south of the specified lat.
        long:   (float) Longitude to split on. If None, won't split east/west
        long_direction: (string) Split east or west of the specified long.
    """
    def latitude(pt):
        """returns the average latitude of a precinct"""
        return (np.mean([shape(pt['geometry']).bounds[1],
                         shape(pt['geometry']).bounds[3]]))

    def longitude(pt):
        """returns the average longitute of a precinct"""
        return (np.mean([shape(pt['geometry']).bounds[0],
                         shape(pt['geometry']).bounds[2]]))

    subset = data
    if lat_north:
        subset = [pt for pt in subset if latitude(pt)<=lat_north]

    if lat_south:
        subset = [pt for pt in subset if latitude(pt)>=lat_south]

    if long_east:
        subset = [pt for pt in subset if longitude(pt)<=long_east]

    if long_west:
        subset = [pt for pt in subset if longitude(pt)>=long_west]

    return subset


def threshold(votes):
    """
    Calculates the threshold number of votes needed to win the election.
    ---
    Args:
        votes: (int) total votes cast in the election
    """
    half = votes/2
    if round(half) == half:
        thresh = (half) + 1
    else:
        thresh = round(half)
    return int(thresh)


def wasted_votes(votes, thresh):
    """
    Calculates how many votes a single party wasted.
    ---
    Args:
        votes: (int) votes for that party
        thresh: (int) threshold of vote needed to win
    """
    if votes<thresh:
        return votes
    else:
        return votes-thresh
