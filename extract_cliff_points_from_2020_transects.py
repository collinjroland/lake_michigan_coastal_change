"""
# CliffDelineaToolPy v1.2.0
# An algorithm to map coastal cliff base and top from topography
# https://github.com/zswirad/CliffDelineaTool
# Zuzanna M Swirad (zswirad@ucsd.edu), Scripps Institution of Oceanography, \
# UC San Diego
# Help in debugging: George Thomas
# Originally coded in MATLAB 2019a (v1.2.0; 2021-11-24)
# Last updated on 2022-1-24 (Python 3.8)
# Modified by Collin Roland 2024-03-31
# To do: Improve functionalization and parallelize
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
import shapely
import statsmodels.api as sm
import sys
import time

np.seterr(divide='ignore', invalid='ignore')
# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

# %% Create project directories

start_time = time.time() # Record time of processing start
os.chdir(homestr + r'\PythonScripts\lake_michigan_dod')
from extract_cliff_point_functions import * # Load helper functions

# Define data path, project path, and name of dataset
data_path = homestr + r"\lake_michigan_bluff_delineation\2020_noaa_nearshore_2\delineation_points_text"
proj_dir = homestr + r"\lake_michigan_bluff_delineation\2020_noaa_nearshore_2\cliff_delineation_params1"
dataset_name = "2020_p1"

# Create new project directory
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
    
# Create subdiretory for logging the console output
log_dir = os.path.join(proj_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
else:
    print("Project log directory exists")
    
# Create log file  
logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M") + '.txt'
logfilename = os.path.join(log_dir, logfilename)
sys.stdout = Logger(logfilename)

# Create output subdirectories:
bluff_top_dir = os.path.abspath(os.path.join(proj_dir, 'delineation_top_points'))
if not os.path.exists(bluff_top_dir):
    os.makedirs(bluff_top_dir)
else:
    print("Bluff top directory exists")
bluff_base_dir = os.path.abspath(os.path.join(proj_dir, 'delineation_base_points'))
if not os.path.exists(bluff_base_dir):
    os.makedirs(bluff_base_dir)
else:
    print("Bluff base directory exists")
# %% Define cliff crest/toe delineation parameters

local_scale = 7 # How many adjacent points to consider as a local scale?
base_max_elev = 179.0 # What is the top limit for cliff base elevation (m)?
base_sea_slope = 10  # What is the max seaward slope for cliff base (deg)?
base_land_slope = 20  # What is the min landward slope for cliff base (deg)?
top_sea_slope = 15  # What is the min seaward slope for cliff top (deg)?
top_land_slope = 10  # What is the max landward slope for cliff top (deg)?
prop_convex = 0.15  # What is the minimal proportion of the distance from trendline #2 to replace modelled cliff top location?
smooth_window = 5  # What is the alongshore moving window (number of transects) for cross-shore smoothing (points)? INTEGER
proj_crs = 'EPSG:6345'

# Create strict parameter dictionary
params_strict = {"local_scale": local_scale, 
          "base_max_elev": base_max_elev,
          "base_sea_slope": base_sea_slope, 
          "base_land_slope": base_land_slope,
          "top_sea_slope": top_sea_slope, 
          "top_land_slope": top_land_slope,
          "prop_convex": prop_convex, 
          "smooth_window": smooth_window, 
          "proj_crs": proj_crs}

# Create relaxed parameter dictionary in case of failed top ID
params_lax = {"local_scale": local_scale, 
          "base_max_elev": base_max_elev,
          "base_sea_slope": base_sea_slope, 
          "base_land_slope": base_land_slope - 2,
          "top_sea_slope": top_sea_slope - 10, 
          "top_land_slope": top_land_slope + 10,
          "prop_convex": prop_convex, 
          "smooth_window": smooth_window, 
          "proj_crs": proj_crs}

# Create parameter dictionary for directory structure
params_etc = {"data_path": data_path,
              "bluff_top_dir": bluff_top_dir,
              "bluff_base_dir": bluff_base_dir}

# Write parameters to log file
params_write = dict(params_strict)
params_write.update(params_etc)
save_default_params(params_write, logfilename)
# %% Load tiles and identify overlapping tiles

os.chdir(home)
tiles_2012 = gpd.read_file(r'.\JABLTX_2012\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2012_edit.shp')
tiles_2020 = gpd.read_file(r'.\JABLTX_2020\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2020.shp')
tiles_2012 = tiles_2012.to_crs(proj_crs)
tiles_2020 = tiles_2020.to_crs(proj_crs)
tiles_2020_merge = shapely.unary_union(tiles_2020.geometry)
tiles_olap = tiles_2012[shapely.intersects(tiles_2012.geometry, tiles_2020_merge).values]
# %% Loop over transect files
colnames =  ['PointID', 'TransectID', 'Elevation', 'Distance', 'Easting', 'Northing']
clip = True
lakeward_limit = 500 - 60
landward_limit = 500 + 150
os.chdir(data_path)
for count, filename in enumerate( glob.glob('*.txt') ):
    if count >= 46:
        # count = 1
        # filename = glob.glob('*.txt')[count]
        start_file = time.time()
        clip = True
        delineate_cliff_features_wrap(clip, lakeward_limit, landward_limit, data_path, filename, bluff_top_dir, bluff_base_dir, colnames, params_strict, params_lax)
        print("File ", filename, ' processed in ', time.time() - start_file, ' seconds.')

end_time = time.time()
print("All files processed in " ,end_time - start_time, " seconds.")
# %% Loop over transect files with parallelization
# colnames =  ['PointID', 'TransectID', 'Elevation', 'Distance', 'Easting', 'Northing']
# clip = True
# lakeward_limit = 500 - 60
# landward_limit = 500 + 150
# clip = True

# os.chdir(data_path)

# import joblib
# start_time = time.time()

# def f(count):
#     start_file = time.time()
#     os.chdir(data_path)
#     filename = glob.glob('*.txt')[count]
#     delineate_cliff_features_wrap(clip, lakeward_limit, landward_limit, data_path, filename, bluff_top_dir, bluff_base_dir, colnames, params_strict, params_lax)
#     print("File ", filename, ' processed in ', time.time() - start_file, ' seconds.')

# joblib.Parallel(n_jobs = -1, prefer = "threads")(joblib.delayed(f)(count) for count in range(0, len( tiles_olap) ) )

# end_time = time.time()
# print("All files processed in " , end_time - start_time, " seconds.")

# %% Merge points

all_top_points = pd.DataFrame()
os.chdir(bluff_top_dir)
for number, fileName in enumerate(glob.glob('*.shp')):
    top_points = gpd.read_file(fileName)
    all_top_points = pd.concat([all_top_points, top_points])
os.chdir(os.path.abspath(os.path.join(bluff_top_dir, r'..')))
all_top_points.to_file(f'{dataset_name}_top_points.shp')

all_base_points = pd.DataFrame()
os.chdir(bluff_base_dir)
for number, fileName in enumerate(glob.glob('*.shp')):
    base_points = gpd.read_file(fileName)
    all_base_points = pd.concat([all_base_points, base_points])
os.chdir(os.path.abspath(os.path.join(bluff_base_dir, r'..')))
all_base_points.to_file(f'{dataset_name}_base_points.shp')
