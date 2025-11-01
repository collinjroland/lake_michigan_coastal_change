# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:03:16 2024

@author: quatlab

# Title: convert_probability_outputs.py
# Author: Collin Roland
# Date Created: 20241118
# Summary: Converts change probability rasters created using Phil Wernette's 
change probability dsmchange.exe approach from ENVI to TIF format
# Date Last Modified: 20250702
# To do:  Nothing
"""
# %% Import packages

from osgeo import gdal
import glob
import joblib
import numpy as np
import os
import pathlib
from pathlib import Path
import rasterio as rio

# %% Self-defined functions

def read_paths(path,extension):
    """Read the paths of all files in a directory (including subdirectories)
    with a specified extension
    Parameters
    -----
    file: (string) path to input file to read
    extension: (string) file extension of interest
    """
    AllPaths = []
    FileNames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                FileNames.append(file)
                filepath = subdir+os.sep+file
                AllPaths.append(filepath)
    return(AllPaths,FileNames)

# %% Set directories

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

change_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\outputs'
output_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\outputs_tif'
output_change_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\outputs_change_tif'


# %% Convert probability rasters to tif

output_fps, output_fns = read_paths(change_dir, 'probability.dat')

def convert_outputs(count, output_fns, output_dir):
    name = os.path.join( output_dir, str( os.path.splitext( os.path.basename(output_fns[count] ) )[0] + '.tif' ) )
    opts = gdal.TranslateOptions(format = "GTiff",
                             callback = gdal.TermProgress_nocb, 
                             creationOptions = ["COMPRESS=LZW", "SPARSE_OK=TRUE", "BIGTIFF=NO"] )
    gdal.Translate(name, output_fps[count] , options = opts)


joblib.Parallel(n_jobs  = 6, prefer = "threads")(joblib.delayed(convert_outputs)(count, output_fns, output_dir) for count in range(0, len(output_fns) ) )            
# %% Convert change rasters to tif

output_fps, output_fns = read_paths(change_dir, '.dat')
output_fps = [i for i in output_fps if 'probability' not in i]
output_fns = [i for i in output_fns if 'probability' not in i]

def convert_outputs(count, output_fns, output_dir):
    name = os.path.join( output_dir, str( os.path.splitext( os.path.basename(output_fns[count] ) )[0] + '.tif' ) )
    opts = gdal.TranslateOptions(format = "GTiff",
                             callback = gdal.TermProgress_nocb, 
                             creationOptions = ["COMPRESS=LZW", "SPARSE_OK=TRUE", "BIGTIFF=NO"] )
    gdal.Translate(name, output_fps[count] , options = opts)


joblib.Parallel(n_jobs  = 6, prefer = "threads")(joblib.delayed(convert_outputs)(count, output_fns, output_change_dir) for count in range(0, len(output_fns) ) )  
# %% Rewrite projection (CRS) information

dem_2020_root = homestr + r'\JABLTX_2020\DEM_NOAA\dem_coreg_to_2012_tiles'
[dem_paths_2020, dem_filenames_2020] = read_paths(dem_2020_root,'.tif')
example_dem = rio.open(dem_paths_2020[0])

change_fps, change_fns = read_paths( output_change_dir, '.tif' )
probability_fps, probability_fns = read_paths( output_dir, '.tif' )

for fp in change_fps:
    with rio.open(fp, 'r+') as rds:
        rds.crs = example_dem.crs
        
for fp in probability_fps:
    with rio.open(fp, 'r+') as rds:
        rds.crs = example_dem.crs  