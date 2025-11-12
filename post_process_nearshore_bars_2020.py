# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:57:41 2024

# Title: post_process_nearshore_bars.py
# Author: Collin Roland
# Date Created: 20241231
# Summary: 
# Date Last Modified: 20250104
# To do: Nothing
"""
# %% Import packages


import geopandas as gpd
import folium
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
import sys
import time
import webbrowser

np.seterr(divide = 'ignore', invalid = 'ignore')
%matplotlib qt5
# %% Self-defined functions
class Logger(object):
    def __init__(self, logfilename):
        self.terminal = sys.stdout
        self.log = open(logfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def save_default_params(params, loggerfilename):

    json_object = json.dumps(params, indent=4)

    with open(loggerfilename, "w") as f:
        f.write(json_object)
        
def cluster_shapes_by_distance(gdf, distance_threshold, check_crs = False):
    """
    Make groups for all shapes within a defined distance. For a shape to be 
    excluded from a group, it must be greater than the defined distance
    from *all* shapes in the group.
    Distances are calculated using shape centroids.

    Parameters
    ----------
    gdf : GeoPandas GeoDataFrame
        A GeoPandas GeoDataFrame of points. Should be a projected CRS where the
        unit is in meters. 
    distance_threshold : float
        Maximum distance between elements, in meters.
    check_crs : bool
        Confirm that the CRS of the geopandas dataframe is projected. This 
        function should not be run with lat/lon coordinates. 

    Returns
    -------
    np.array
        Array of numeric labels assigned to each row in gdf.

    """
    if check_crs:
        assert gdf.crs.is_projected, 'geodf should be a projected crs with meters as the unit'
    points_xy = [ [point.x, point.y] for point in gdf.geometry ]
    cluster = AgglomerativeClustering(n_clusters = None, 
                                      linkage = 'single',
                                      metric = 'euclidean',
                                      distance_threshold = distance_threshold)
    cluster.fit(points_xy)
    return cluster.labels_
# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')
# %% Make output directories

proj_dir = homestr + r'\lake_michigan_bluff_delineation\2020_noaa_nearshore\shoreline_extract_param2'
dataset_name = '2020'

# For logging the console output
log_dir = os.path.join(proj_dir, 'logs_post_process')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
else:
    print("Project log directory exists")
logfilename = 'log_' + time.strftime("%Y-%m-%d_%H%M") + '.txt'
logfilename = os.path.join(log_dir, logfilename)
sys.stdout = Logger(logfilename)

# Create output subdirectories: 
peaks_dir = os.path.abspath(os.path.join(proj_dir, 'peak_points_post_process'))
if not os.path.exists(peaks_dir):
    os.makedirs(peaks_dir)
else:
    print("Bar peaks points directory exists")
troughs_dir = os.path.abspath(os.path.join(proj_dir, 'trough_points_post_process'))
if not os.path.exists(troughs_dir):
    os.makedirs(troughs_dir)
else:
    print("Bar troughs points directory exists")
    
# Define the input parameters for shoreline and bar/trough identification:
params = {"cluster_threshold": 20,
          "distance_threshold": 11.0,
          "scatter_threshold_factor": 0.1} 

params_write = dict(params)
save_default_params(params_write, logfilename)
# %% Read in peaks and troughs
os.chdir(proj_dir)
all_peaks = gpd.read_file( glob.glob('*_peaks.gpkg')[0] )
all_troughs = gpd.read_file( glob.glob('*_troughs.gpkg')[0] )
# %% Filtering peaks and troughs to only those transects with both a peak
# and trough
lowess = sm.nonparametric.lowess

tile_names = all_peaks['tile_name'].unique()
for tile_name in tile_names:
    # Debugging
    # tile_name = '2012_NCMP_IN_Michigan_04_BareEarth_1mGrid'
    tile_peaks = all_peaks.loc[ all_peaks['tile_name'] == tile_name]
    tile_peaks_orig = tile_peaks.copy()
    tile_peaks['x'] = tile_peaks.geometry.x
    tile_peaks['y'] = tile_peaks.geometry.y
    tile_troughs = all_troughs.loc[ all_troughs['tile_name'] == tile_name]
    tile_troughs_orig = tile_troughs.copy()
    tile_troughs['x'] = tile_troughs.geometry.x
    tile_troughs['y'] = tile_troughs.geometry.y
    tile_peaks['group'] = cluster_shapes_by_distance(tile_peaks, distance_threshold = params['distance_threshold'] ) # get clusters of points
    tile_troughs['group'] = cluster_shapes_by_distance(tile_troughs, distance_threshold = params['distance_threshold'] ) # get clusters of points
    tile_peaks = tile_peaks.groupby('group').filter(lambda x: len(x) > params['cluster_threshold'] ) # get points in clusters that exceed the cluster threshold for number of points
    tile_troughs = tile_troughs.groupby('group').filter(lambda x: len(x) > params['cluster_threshold'] ) # get points in clusters that exceed the cluster threshold for number of points
    peak_groups_to_keep = []
    # Plotting to debug
    # fig, ax = plt.subplots(1)
    # tile_peaks.plot('group', cmap = 'tab20c', legend = True, ax = ax )
    # for x, y, label in zip(tile_peaks.geometry.x, tile_peaks.geometry.y, tile_peaks.group):
    #     ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    for group in tile_peaks['group'].unique():
        cluster = tile_peaks.loc[ tile_peaks['group'] == group]
        lowess_rmse = np.mean( np.sqrt( ( cluster.x.values - lowess(cluster.x, cluster.y, frac = 0.2, return_sorted = False) ) ** 2 ) )
        # Debugging
        # lowess_results = lowess(cluster.x, cluster.y, frac = 0.1)
        # test = cluster.x.values - lowess(cluster.x, cluster.y, frac = 0.2)[:, 1]
        # fig, ax = plt.subplots(1)
        # ax.scatter(cluster.x, cluster.y)
        # ax.plot(lowess_results[:, 1], lowess_results[:, 0], color = 'k')
        if lowess_rmse < ( len(cluster) * params['scatter_threshold_factor'] ) :
            peak_groups_to_keep.append( group )
    trough_groups_to_keep = []
    for group in tile_troughs['group'].unique():
        cluster = tile_troughs.loc[ tile_troughs['group'] == group]
        lowess_rmse = np.mean( np.sqrt( ( cluster.x.values - lowess(cluster.x, cluster.y, frac = 0.2, return_sorted = False ) ) ** 2 ) )
        if lowess_rmse < ( len(cluster) * params['scatter_threshold_factor'] ) :
            trough_groups_to_keep.append( group )
    tile_peaks = tile_peaks.loc[ tile_peaks['group'].isin(peak_groups_to_keep)]
    tile_troughs = tile_troughs.loc[ tile_troughs['group'].isin(trough_groups_to_keep)]
    tile_peaks.to_file(os.path.join( peaks_dir, str( tile_name + '_peaks_pp.gpkg') ) )
    tile_troughs.to_file(os.path.join( troughs_dir, str( tile_name + '_troughs_pp.gpkg') ) )
    # Plotting
    # test_map = folium.Map()
    # tile_peaks.explore(m = test_map, color = 'blue', name = 'post_processed peaks')
    # tile_troughs.explore(m = test_map, color = 'blue', name = 'post-processed troughs')
    # tile_peaks_orig.explore(m = test_map, color = 'red', name = 'peaks')
    # tile_troughs_orig.explore(m = test_map, color = 'red', name = 'troughs')
    # folium.LayerControl().add_to(test_map)
    # test_map.save('test_map.html')
    # webbrowser.open('test_map.html')

# %% Merge postprocessed outputs

os.chdir( peaks_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_bar_peaks_pp.gpkg')

os.chdir( troughs_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_bar_troughs_pp.gpkg')
# %% Extract points that were removed by filtering

os.chdir( peaks_dir )
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
filtered_peaks = gpd.read_file(f'lake_michigan_{dataset_name}_bar_peaks_pp.gpkg')
os.chdir( troughs_dir )
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
filtered_troughs = gpd.read_file(f'lake_michigan_{dataset_name}_bar_troughs_pp.gpkg')

peaks_in_filter_idx = filtered_peaks.sindex.query(all_peaks.geometry, predicate="intersects")[0]
all_peaks_not_in_filter = all_peaks.loc[~all_peaks.index.isin(peaks_in_filter_idx)]
all_peaks_not_in_filter.to_file(f'lake_michigan_{dataset_name}_bar_peaks_pp_not.gpkg')

troughs_in_filter_idx = filtered_troughs.sindex.query(all_troughs.geometry, predicate="intersects")[0]
all_troughs_not_in_filter = all_troughs.loc[~all_troughs.index.isin(troughs_in_filter_idx)]
all_troughs_not_in_filter.to_file(f'lake_michigan_{dataset_name}_bar_troughs_pp_not.gpkg')
# %% Remove duplicates after filtering

all_peaks_manual_edit = gpd.read_file(r'D:\CJR\lake_michigan_bluff_delineation\manual_new_transects\manual_edit_points\2020_bar_peaks_pp_manual_edit.gpkg')
all_peaks_manual_edit_no_dups = all_peaks_manual_edit.drop_duplicates(subset = 'geometry')
all_peaks_manual_edit_no_dups.to_file(r'D:\CJR\lake_michigan_bluff_delineation\manual_new_transects\manual_edit_points\2020_bar_peaks_pp_manual_edit_no_dups.gpkg')
all_troughs_manual_edit = gpd.read_file(r'D:\CJR\lake_michigan_bluff_delineation\manual_new_transects\manual_edit_points\2020_bar_troughs_pp_manual_edit.gpkg')
all_troughs_manual_edit_no_dups = all_troughs_manual_edit.drop_duplicates(subset = 'geometry')
all_troughs_manual_edit_no_dups.to_file(r'D:\CJR\lake_michigan_bluff_delineation\manual_new_transects\manual_edit_points\2020_bar_troughs_pp_manual_edit_no_dups.gpkg')
