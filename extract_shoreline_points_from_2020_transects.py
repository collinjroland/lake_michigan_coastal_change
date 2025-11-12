# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:11:17 2024

@author: colli

# Title: extract_shoreline_points_from_2020_transects.py
# Author: Collin Roland
# Date Created: 20241122
# Summary:  Extracts shoreline points and nearshore bar peaks/troughs from
shore perpendicular transects with 2020 elevation data
# Date Last Modified: 20250114
# To do: None
"""
# %% Import packages

import geopandas as gpd
import glob
import json
import math
import numpy as np
import os
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import sys
import time

np.seterr(divide = 'ignore', invalid = 'ignore')
# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

# %% Import tiles to define files to be looped over (only if there are some txt files you want to skip and you want to filter based on overlapping tile names)

# import shapely
# os.chdir(home)
# tiles2012 = gpd.read_file(r'.\JABLTX_2012\gl2012_usace_lakemichigan_index.shp')
# tiles2012 = tiles2012.to_crs('EPSG:32616')
# tiles2020 = gpd.read_file(r'.\JABLTX_2020\usace2020_lake_mich_il_in_mi_wi_index.shp')
# tiles2020 = tiles2020.to_crs('EPSG:32616')
# tiles2020_merge = shapely.unary_union(tiles2020.geometry)
# tiles_olap = tiles2012[shapely.intersects(tiles2012.geometry, tiles2020_merge).values]

# %% Create output directories and define parameterization
os.chdir(homestr + r'\PythonScripts\lake_michigan_dod')
from extract_shoreline_functions import *

data_path = homestr + r'\lake_michigan_bluff_delineation\2020_noaa_nearshore\delineation_points_text'
proj_dir = homestr + r'\lake_michigan_bluff_delineation\2020_noaa_nearshore\shoreline_extract_param4'
dataset_name = "2020"


# Create new project directory
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
# For logging the console output
log_dir = os.path.join(proj_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
else:
    print("Project log directory exists")
logfilename = 'log_' + time.strftime("%Y-%m-%d_%H%M") + '.txt'
logfilename = os.path.join(log_dir, logfilename)
sys.stdout = Logger(logfilename)

# Create output subdirectories:
shorelines_mean_dir = os.path.abspath(os.path.join(proj_dir, 'shoreline_mean_points'))
if not os.path.exists(shorelines_mean_dir):
    os.makedirs(shorelines_mean_dir)
else:
    print("Shoreline mean points directory exists")
shorelines_dir = os.path.abspath(os.path.join(proj_dir, 'shoreline_points'))
if not os.path.exists(shorelines_dir):
    os.makedirs(shorelines_dir)
else:
    print("Shoreline points directory exists")   
peaks_dir = os.path.abspath(os.path.join(proj_dir, 'peak_points'))
if not os.path.exists(peaks_dir):
    os.makedirs(peaks_dir)
else:
    print("Bar peaks points directory exists")
troughs_dir = os.path.abspath(os.path.join(proj_dir, 'trough_points'))
if not os.path.exists(troughs_dir):
    os.makedirs(troughs_dir)
else:
    print("Bar troughs points directory exists")
    
# Define the input parameters for shoreline and bar/trough identification:
params = {"shoreline_elevation": 177.31,
          "shoreline_elevation_mean": 177.31,
          "peak_lower_height": 171,
          "peak_upper_height": 177,
          "peak_prominence": 0.2,
          "trough_elevation_threshold": 177.4,
          "crs": "EPSG:6345"}
colnames = ['num_transect', 'peak_heights', 'prominences']


params_etc = {"data_path": data_path,
              "shoreline_mean_dir": shorelines_mean_dir,
              "shoreline_dir": shorelines_dir,
              "bar_peaks_dir": peaks_dir,
              "bar_troughs_dir": troughs_dir}

params_write = dict(params)
params_write.update(params_etc)
save_default_params(params_write, logfilename)

# %% Loop over transect data files (each tile)

# Debugging
# tile_name = '2012_NCMP_IL_Michigan_02_BareEarth_1mGrid_points.txt'


start_time = time.time()
os.chdir(data_path)
for count, filename in enumerate( glob.glob('*.txt') ):
    if count > 43:
        # if (number >= 0) & any(fileName[14:-11] in x for x in tiles_olap['Name'].values):  # Use this line to filter to only overlapping tiles
        filename = (glob.glob('*.txt'))[count]
        # Debugging
        # if filename == tile_name:
        # Main
        start_file = time.time()
        extract_shoreline_points_wrap(data_path, filename, shorelines_dir,shorelines_mean_dir, peaks_dir, troughs_dir, colnames, params)

end_time = time.time()
print("All files processed in " ,end_time - start_time, " seconds.")
# %% Loop over transect data files (each tile) (parallelized) ( does not seem to work)
# os.chdir(data_path)
# import joblib
# start_time = time.time()

# def f(count, shorelines_dir, shorelines_mean_dir, peaks_dir, troughs_dir, colnames, params):
#     start_file = time.time()
#     os.chdir(data_path)
#     filename = glob.glob('*.txt')[count]
#     extract_shoreline_points_wrap(data_path, filename, shorelines_dir,
#                                       shorelines_mean_dir, peaks_dir, troughs_dir, colnames,
#                                       params)
        
# joblib.Parallel(n_jobs = -1, prefer = "threads")(joblib.delayed(f)(count, shorelines_dir, shorelines_mean_dir, peaks_dir, troughs_dir, colnames, params) for count in range(0, len( glob.glob( '*.txt') ) ) )

# end_time = time.time()
# print("All files processed in " ,end_time - start_time, " seconds.")
# %% Merge output points

os.chdir( shorelines_mean_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_shorelines_mean.gpkg')

os.chdir( shorelines_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_shorelines.gpkg')

os.chdir( peaks_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_bar_peaks.gpkg')

os.chdir( troughs_dir )
all_features = pd.DataFrame()
for number, fileName in enumerate( glob.glob('*.gpkg') ):
    features = gpd.read_file(fileName)
    all_features = pd.concat([all_features, features])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_features.to_file(f'lake_michigan_{dataset_name}_bar_troughs.gpkg')
    