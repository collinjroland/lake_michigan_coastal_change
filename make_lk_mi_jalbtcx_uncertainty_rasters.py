# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:50:57 2024

@author: quatlab

# Title: make_lk_mi_jalbtcx_uncertainty_rasters.py
# Author: Collin Roland
# Date Created: 20241116
# Summary: Create unertainty rasters for 2012 and 2020 Lake Michigan
JALBTCX rasters
# Date Last Modified: 20250702
# To do:  Nothing
"""
# %% Import packages

from osgeo import gdal
import geopandas as gpd
import glob
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from pathlib import Path
import rasterio
from rasterio import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import errors
import shapely
from shapely import buffer, LineString, Point, distance
%matplotlib qt5
# %% Self-defined functions

def read_file(file):
    """Read in a raster file
    
    Parameters
    -----
    file: (string) path to input file to read
    """
    return(rasterio.open(file))


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

# %% Set home directory and read filepaths/names

homestr = r'D:\CJR'
home = Path(r'D:\CJR')


os.chdir(home)
tiles_2012 = gpd.read_file(r'.\JABLTX_2012\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2012_edit.shp')
tiles_2020 = gpd.read_file(r'.\JABLTX_2020\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2020.shp')

tiles_2012 = tiles_2012.to_crs('EPSG:6345')
tiles_2020 = tiles_2020.to_crs('EPSG:6345')

tiles_2012_filter = []
for count in range( 0, len( tiles_2012) ):
    # Debug
    # count = 0
    tile_2012 = tiles_2012.iloc[count]
    filt = tile_2012.geometry.intersects(tiles_2020.geometry)
    if True in filt.values:
        tiles_2012_filter.append(True)
    else:
        tiles_2012_filter.append(False)
        

tiles_2012_clip = tiles_2012.loc[tiles_2012_filter]

dem_2012_root = homestr + r'\JABLTX_2012\DEM_NOAA\dem_project_6345_1m'
dem_2020_root = homestr + r'\JABLTX_2020\DEM_NOAA\dem_coreg_to_2012_tiles'

[dem_paths_2012, dem_filenames_2012] = read_paths(dem_2012_root,'.tif')
[dem_paths_2020, dem_filenames_2020] = read_paths(dem_2020_root,'.tif')
# %% Unmask 2012 DEMs

# Create a directory for unamsked 2012 DEM rasters
proj_dir = homestr + r'\JABLTX_2012\DEM_NOAA\dem_project_6345_unmask'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2012_unmask_root = proj_dir

for count in range(0 , len( dem_paths_2012 ) ):
    # count = 0
    basename = os.path.splitext( os.path.basename( dem_filenames_2012[count] ) )[0]
    template = basename
    template = template.removesuffix('_1m')
    template = template.removesuffix('_fugro')
    if ( ( str(template + '.tif') in tiles_2012_clip.location.values) == True):
        dem = read_file(dem_paths_2012[count])
        dst_kwargs = dem.meta.copy()
        dst_kwargs.update({"compress":"LZW",
                           "dtype":"float32"})
        dem_arr = dem.read(1)
        name = os.path.join(dem_2012_unmask_root, str(basename + r'_unmask.tif') )
        with rasterio.open(name, "w", **dst_kwargs) as dst:
                dst.write(dem_arr, indexes = 1)

## Translate *.tif's to ENVI format

# Create a directory for unamsked 2012 DEM rasters in ENVI format
proj_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2012'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2012_unmask_envi_root = proj_dir

[FilePaths, FileNames] = read_paths(dem_2012_unmask_root, '.tif')    
for count, value in enumerate(FilePaths):
    if count == 0:
        name = os.path.join( dem_2012_unmask_envi_root, str( os.path.splitext( os.path.basename(FileNames[count] ) )[0] + '.dat' ) )
        opts = gdal.TranslateOptions(format = 'ENVI')
        gdal.Translate(name, FilePaths[count] , options = opts)
# %% Generate 2012 uncertainty rasters

# Create a directory for 2012 DEM uncertainty
proj_dir = homestr + r'\JABLTX_2012\DEM_NOAA\dem_project_6345_uncertainty'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2012_uncert_root = proj_dir

[dem_paths_2012, dem_filenames_2012] = read_paths(dem_2012_unmask_root, '.tif')    

for count in range(0 , len( dem_paths_2012 ) ):
    basename = os.path.splitext( os.path.basename( dem_filenames_2012[count] ) )[0]
    dem = read_file(dem_paths_2012[count])
    dst_kwargs = dem.meta.copy()
    dst_kwargs.update({"compress":"LZW",
                       "dtype":"float32"})
    dem_arr = dem.read(1)
    # err_arr = dem_arr.copy()
    # err_arr[ dem_arr > 0] = err_arr[dem_arr > 0] ** 0 * 0.15
    dep_arr = dem_arr.copy()
    dep_arr[dep_arr > 0] = dep_arr[dep_arr > 0 ] - 175.9
    err_arr = dep_arr.copy()
    err_arr[ (dep_arr < -2) & (dep_arr > -100) ] = np.sqrt( (0.3 ** 2) + ( (0.013 * err_arr[ (dep_arr < -2) & (dep_arr > -100) ]) ** 2) )
    err_arr[ (dep_arr >= -2) & (dep_arr < 0) ] = np.sqrt( (0.25 ** 2) + ( (0.0075 * err_arr[ (dep_arr >= -2) & (dep_arr < 0) ]) ** 2) )
    err_arr[ (dep_arr > 0) ] = ( err_arr[ (dep_arr > 0) ] ** 0 ) * 0.2
    
    name = os.path.join(dem_2012_uncert_root, str(basename + r'_uncert.tif') )
    with rasterio.open(name, "w", **dst_kwargs) as dst:
            dst.write(err_arr, indexes = 1)

## Translate *.tif's to ENVI format

# Create a directory for unamsked 2020 DEM rasters in ENVI format
proj_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2012_uncert'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2012_uncert_envi_root = proj_dir

[FilePaths, FileNames] = read_paths(dem_2012_uncert_root, '.tif')    
for count, value in enumerate(FilePaths):
    name = os.path.join( dem_2012_uncert_envi_root, str( os.path.splitext( os.path.basename(FileNames[count] ) )[0] + '.dat' ) )
    opts = gdal.TranslateOptions(format = 'ENVI')
    gdal.Translate(name, FilePaths[count] , options = opts)            
# %% Unmask 2020 DEMs

# Create a directory for unamsked 2020 DEM rasters
proj_dir = homestr + r'\JABLTX_2020\DEM_NOAA\dem_coreg_to_2012_tiles_unmask'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2020_unmask_root = proj_dir

import joblib
def unmask_rasters(count, dem_filenames_2020, dem_paths_2020, dem_2020_unmask_root):
    basename = os.path.splitext( os.path.basename( dem_filenames_2020[count] ) )[0]
    dem = read_file(dem_paths_2020[count])
    dst_kwargs = dem.meta.copy()
    dst_kwargs.update({"compress":"LZW",
                       "dtype":"float32"})
    dem_arr = dem.read(1)
    name = os.path.join(dem_2020_unmask_root, str(basename + r'_unmask.tif') )
    with rasterio.open(name, "w", **dst_kwargs) as dst:
            dst.write(dem_arr, indexes = 1)
 
joblib.Parallel(n_jobs  = 6, prefer = "threads")(joblib.delayed(unmask_rasters)(count, dem_filenames_2020, dem_paths_2020, dem_2020_unmask_root) for count in range(0, 73) )            
# %%  Translate 2020 unmasked *.tif's to ENVI format

# Create a directory for unamsked 2020 DEM rasters in ENVI format
proj_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2020'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2020_unmask_envi_root = proj_dir

[FilePaths, FileNames] = read_paths(dem_2020_unmask_root, '.tif')    
for count, value in enumerate(FilePaths):
    # count = 55
    # value = FilePaths[count]
    name = os.path.join( dem_2020_unmask_envi_root, str( os.path.splitext( os.path.basename(FileNames[count] ) )[0] + '.dat' ) )
    opts = gdal.TranslateOptions(format = 'ENVI')
    gdal.Translate(name, FilePaths[count] , options = opts)
# %% Generate 2020 uncertainty rasters

# Create a directory for 2020 DEM uncertainty
proj_dir = homestr + r'\JABLTX_2020\DEM_NOAA\dem_coreg_to_2012_tiles_uncertainty'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2020_uncert_root = proj_dir

def generate_uncertainty_rasters( count, dem_paths_2020, dem_filenames_2020, dem_2020_uncert_root):
    basename = os.path.splitext( os.path.basename( dem_filenames_2020[count] ) )[0]
    dem = read_file(dem_paths_2020[count])
    dst_kwargs = dem.meta.copy()
    dst_kwargs.update({"compress":"LZW",
                       "dtype":"float32"})
    dem_arr = dem.read(1)
    dep_arr = dem_arr.copy()
    dep_arr[dep_arr > 0] = dep_arr[dep_arr > 0 ] - 177.3
    err_arr = dep_arr.copy()
    err_arr[ (dep_arr < -2) & (dep_arr > -100) ] = np.sqrt( (0.3 ** 2) + ( (0.013 * err_arr[ (dep_arr < -2) & (dep_arr > -100) ]) ** 2) )
    err_arr[ (dep_arr >= -2) & (dep_arr < 0) ] = np.sqrt( (0.25 ** 2) + ( (0.0075 * err_arr[ (dep_arr >= -2) & (dep_arr < 0) ]) ** 2) )
    err_arr[ (dep_arr > 0) ] = ( err_arr[ (dep_arr > 0) ] ** 0 ) * 0.2
    name = os.path.join(dem_2020_uncert_root, str(basename + r'_uncert.tif') )
    with rasterio.open(name, "w", **dst_kwargs) as dst:
            dst.write(err_arr, indexes = 1)
            
joblib.Parallel(n_jobs  = 6, prefer = "threads")(joblib.delayed(generate_uncertainty_rasters)(count, dem_paths_2020, dem_filenames_2020, dem_2020_uncert_root) for count in range(0, len(dem_paths_2020) ) )            

            
# %% Translate 2020 uncertainty *.tif's to ENVI format

# Create a directory for unamsked 2020 DEM rasters in ENVI format
proj_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2020_uncert'
# Create new project
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)
else:
    print("Project  directory exists")
dem_2020_uncert_envi_root = proj_dir

[FilePaths, FileNames] = read_paths(dem_2020_uncert_root, '.tif')    
for count, value in enumerate(FilePaths):
    name = os.path.join( dem_2020_uncert_envi_root, str( os.path.splitext( os.path.basename(FileNames[count] ) )[0] + '.dat' ) )
    opts = gdal.TranslateOptions(format = 'ENVI')
    gdal.Translate(name, FilePaths[count] , options = opts)