# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:00:20 2023

@author: quatlab

# Title: make_lk_mi_nearshore_transects_2020_noaa_refactor.py
# Author: Collin Roland
# Date Created: 20241121
# Summary: Builds transects along a shoreline file and extracts elevation values along the transects from a specified raster
# Date Last Modified: 20241122
# To do: Confirm that it works correctly for 2020 data
"""

# %% Import packages

from osgeo import gdal
import geopandas as gpd
import glob
import math
import numpy as np
import os
import pandas as pd
import pathlib
from pathlib import Path
import pyproj
import rasterio
from rasterio import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import errors
import re
import shapely
from shapely import buffer, LineString, Point, distance
import shapelysmooth
%matplotlib qt5

# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

# %% Self-defined functions


def read_file(file):
    """Read in a raster file using rasterio
    
    Parameters
    -----
    file: (string) path to input file to read
    """
    return(rasterio.open(file))


def reproj_match(infile, match):
    """Reproject a file to match the shape and projection of existing raster. 
    Uses bilinear interpolation for resampling and reprojects to an in-memory
    file.
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )
            # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": match.nodata})
        memfile = MemoryFile()
        # with MemoryFile() as memfile:
        with memfile.open(**dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)
        try:
            data = memfile.open()
            error_state = False
            return data, error_state
        except rasterio.errors.RasterioIOError as err:
            error_state = True
            data= []
            return data,error_state
            pass
            #with memfile.open() as dataset:  # Reopen as DatasetReader
                #return dataset

                
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


def get_cell_size(str_grid_path):
    with rasterio.open(str(str_grid_path)) as ds_grid:
        cs_x, cs_y = ds_grid.res
    return cs_x


def define_grid_projection(str_source_grid, dst_crs, dst_file):
    print('Defining grid projection:')
    with rasterio.open(str_source_grid, 'r') as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
        })
        arr_src = src.read(1)
        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            dst.write(arr_src, indexes=1)


def reproject_grid_layer(str_source_grid, dst_crs, dst_file, resolution, logger):
    # reproject raster plus resample if needed
    # Resolution is a pixel value as a tuple
    try:
        st = timer()
        with rasterio.open(str_source_grid) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Reprojected DEM. Time elapsed: {end} mins')
        return dst_file
    except:
        logger.critical(f'{str_source_grid}: failed to reproject.')
        sys.exit(1)


def reproject_vector_layer(in_shp, str_target_proj4, logger):
    print(f'Reprojecting vector layer: {in_shp}')
    proj_shp = in_shp.parent / f'{in_shp.stem}_proj.shp'
    if proj_shp.is_file():
        logger.info(f'{proj_shp} reprojected file already exists\n')
        return str(proj_shp)
    else:
        gdf = gpd.read_file(str(in_shp))
        # fix float64 to int64
        float64_2_int64 = ['NHDPlusID', 'Shape_Area', 'DSContArea', 'USContArea']
        for col in float64_2_int64:
            try:
                gdf[col] = gdf[col].astype(np.int64)
            except KeyError:
                pass
        gdf_proj = gdf.to_crs(str_target_proj4)
        gdf_proj.to_file(str(proj_shp))
        logger.info(f'{proj_shp} successfully reprojected\n')
        return str(proj_shp)


def clip_features_using_grid(
        str_lines_path, output_filename, str_dem_path, in_crs, logger, mask_shp):
    # clip features using HUC mask, if the mask doesn't exist polygonize DEM
    mask_shp = Path(mask_shp)
    if mask_shp.is_file():
        st = timer()
        # whitebox clip
        WBT.clip(str_lines_path, mask_shp, output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')
    else:
        st = timer()
        logger.warning(f'''
        {mask_shp} does not file exists. Creating new mask from DEM.
        This step can be error prone please review the output.
        ''')
        # Polygonize the raster DEM with rasterio:
        with rasterio.open(str(str_dem_path)) as ds_dem:
            arr_dem = ds_dem.read(1)
        arr_dem[arr_dem > 0] = 100
        mask = arr_dem == 100
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(arr_dem, mask=mask, transform=ds_dem.transform))
            )
        poly = list(results)
        poly_df = gpd.GeoDataFrame.from_features(poly)
        poly_df.crs = in_crs
        # poly_df = poly_df[poly_df.raster_val == 100.0]
        # tmp_shp = os.path.dirname(str_dem_path) + "/mask.shp"  # tmp huc mask
        poly_df.to_file(str(mask_shp))
        # whitebox clip
        WBT.clip(str_lines_path, str(mask_shp), output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')


def open_memory_tif(arr, meta):
    from rasterio.io import MemoryFile
    #     with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS='ALL_CPUS'):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(arr, indexes=1)
        return memfile.open()


def raster_clip(infile, clip_geom, crs_in):
    #Debugging
    # infile= value
    # crs_in = clip_geom.crs
    # clip_geom = clip_geom
    
    infile=infile
    clip_geom = clip_geom.to_crs('EPSG:32616')
    clip_geom = clip_geom.reset_index()
    dem = read_file(infile)
    try:
        out_image, out_transform = mask(dem,[clip_geom.geometry[0]], nodata=dem.meta['nodata'],crop=True)
        error_state = False
        out_meta = dem.meta
        out_meta.update({"crs": dem.crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": out_transform,
                           "width": out_image.shape[2],
                           "height": out_image.shape[1]})
        clip_kwargs = out_meta
        memfile_clip = MemoryFile()
        with memfile_clip.open(**clip_kwargs) as dst:
            dst.write(out_image)
        return memfile_clip,error_state
    except ValueError as err:
        memfile_clip=[]
        error_state=True
        return memfile_clip,error_state
        pass
   


def fix_index(gdf):
    gdf["row_id"] = gdf.index
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_index("row_id", inplace=True)
    return (gdf)


def linestring_to_points(feature, line):
    return {feature: line.coords}


# %% Read NOAA DEM tile indices and filter to those that overlap

# Define paths
os.chdir(home)
tiles_2012 = gpd.read_file(r'.\JABLTX_2012\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2012_edit.shp')
tiles_2020 = gpd.read_file(r'.\JABLTX_2020\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2020.shp')

# Project tiles to DEM coordinate system
tiles_2012 = tiles_2012.to_crs('EPSG:6345')
tiles_2020 = tiles_2020.to_crs('EPSG:6345')

# Clip 2012 tiles to those that intersect 2020 tiles
tiles_2012 = tiles_2012[ tiles_2012.geometry.intersects( shapely.ops.unary_union( tiles_2020.geometry ) ) ]
tiles_2012 = tiles_2012.reset_index(drop = True)
# tiles_2012.to_file(homestr + r'\JABLTX_2012\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2012_edit_clip.shp')
# %% Read input DEMs filenames and paths and set up directory 

dems_2020_root = homestr +  r'\JABLTX_2020\DEM_NOAA\dem_coreg_to_2012_tiles'
[dem_2020_fps, dem_2020_fns] = read_paths(dems_2020_root,'.tif')


# Generate directory structure for outputs
outdir = r"D:\CJR\lake_michigan_bluff_delineation\2020_noaa_nearshore_2" 
# Create directory if it does not exist
if not os.path.exists(outdir):
    os.mkdir(outdir)
else:
    print("Project  directory exists")
os.chdir(outdir)
if not os.path.exists(r'.\Transects'):
    os.mkdir(r'.\Transects')
else:
    print("Transects  directory exists")
if not os.path.exists(r'.\EndPoints'):
    os.mkdir(r'.\EndPoints')
else:
    print("EndPoints  directory exists")
if not os.path.exists(r'.\StartPoints'):
    os.mkdir(r'.\StartPoints')
else:
    print("StartPoints  directory exists")
if not os.path.exists(r'.\Shorelines'):
    os.mkdir(r'.\Shorelines')
else:
    print("Shorelines  directory exists")           
if not os.path.exists(r'.\delineation_points_text'):
    os.mkdir(r'.\delineation_points_text')
else:
    print("delineation_points_text  directory exists")
    
# Read in the shoreline file
shoreline = gpd.read_file(r'D:\CJR\lake_michigan_bluff_delineation\great_lakes_hardened_shorelines_lake_mi_nearshore_edit.shp')
shoreline = shoreline.to_crs('EPSG:6345')

# %%  Editing make transects function
os.chdir(homestr + r'\PythonScripts\lake_michigan_dod')
from make_cross_section_functions import *

# %% Make transects with parallelization
param_dict = {'buffer_dist': 500, "xsec_spacing":5, "poslength":250.0, "neglength":500.0, "simp_tolerance":50.0, "snap_tolerance":20.0, "point_spacing":1.0, "crs":6345}
great_lakes_shoreline_path = r'D:\CJR\lake_michigan_bluff_delineation\great_lakes_hardened_shorelines_lake_mi_nearshore_edit.shp'
tiles_path = homestr + r'\JABLTX_2012\DEM_NOAA\tileindex_USACE_Lake_Mich_IL_IN_MI_WI_DEM_2012_edit_clip.shp'


import joblib
def f(count):
    print('Working on tile ',count,' of', len( tiles_2012 ),' .')
    shoreline_clip, tile_sing_clean, tile_sing, tile_name = prep_shoreline_and_tile(great_lakes_shoreline_path, tiles_path, count)
    merged_dem = make_merged_dem(count, tiles_path, tile_sing, param_dict['buffer_dist'], dems_2020_root, dem_2020_fps)
    gen_xsec_wrap(shoreline_clip, outdir, tile_name, param_dict["poslength"], param_dict["neglength"], param_dict["xsec_spacing"], param_dict["simp_tolerance"], param_dict["point_spacing"], param_dict["snap_tolerance"],merged_dem, crs= param_dict["crs"])
    os.chdir(os.path.join(outdir, 'Transects'))
    search_pattern = str(tile_name + '*.gpkg')
    transect_files = (glob.glob(search_pattern))
    transects = pd.DataFrame()
    for file in transect_files:
        transect = gpd.read_file(file)
        transects = pd.concat([transects, transect]) 
    xsec_points = make_xsec_points(transects, merged_dem)
    merged_dem = None
    xsec_points.to_csv( os.path.join( outdir, 'delineation_points_text', (tile_name + '_points.txt' ) ), index = False)
    merged_dem = None
        
joblib.Parallel(n_jobs = -2, prefer = "threads")(joblib.delayed(f)(i) for i in range(0,  len(tiles_2012) ) ) 
# %% Fix 2020 NOAA nearshore transect ID's

os.chdir(outdir)
if not os.path.exists(r'.\Transects_mod'):
    os.mkdir(r'.\Transects_mod')
else:
    print("Transects_mod  directory exists")
    
os.chdir(os.path.join(outdir, 'Transects'))
transect_filenames = glob.glob('*.gpkg')
for count in range( 0, len( tiles_2012 ) ):
    # Debugging
    # count = 0
    tile_2012_sing = tiles_2012.iloc[[count]]  # pull out a single 2009 tile
    tile_name = os.path.splitext(tile_2012_sing.location.iloc[0])[0]
    search_pattern = str( tile_name + '*.gpkg' )
    transect_files = (glob.glob(search_pattern))
    if len(transect_files) > 0:
        transect_numbers = [i[-9:-5] for i in transect_files]
        transect_numbers = [int( ( re.findall( r'\d+', i) )[0]) for i in transect_numbers]
        transect_df = pd.DataFrame(columns = ['Filenames', 'transect_group'])
        transect_df['Filenames'] = transect_files
        transect_df['transect_group'] = transect_numbers
        transect_df = transect_df.sort_values('transect_group')
        transect_df = transect_df.reset_index()
        transect_df = transect_df.drop('index', axis = 1)
        all_transects = pd.DataFrame()
        for count in range(0, len(transect_df) ):
            transects = gpd.read_file( transect_df['Filenames'][count] )
            all_transects = pd.concat( [all_transects, transects] )
        all_transects = all_transects.reset_index()    
        all_transects = all_transects.drop('index', axis = 1)
        all_transects['FID'] = all_transects.index
        all_transects['tile_name'] = tile_name
        all_transects.to_file(os.path.join( outdir, 'Transects_mod' ,(tile_name + '_transects.gpkg')))

# %% Merge 2020 nearshore transects

os.chdir(os.path.join(outdir,'Transects_mod'))
all_transects = pd.DataFrame()
for number, fileName in enumerate(glob.glob('*.gpkg')):
    transects = gpd.read_file(fileName)
    all_transects = pd.concat([all_transects, transects])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_transects.to_file(f'lake_michigan_2020_nearshore_transects_noaa.gpkg')