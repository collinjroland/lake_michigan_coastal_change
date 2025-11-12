# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:15:06 2024

@author: colli

# Title:extract_shoreline_functions.py
# Author: Collin Roland
# Date Created: 20250101
# Summary: Functions for extracting shoreline positions and bar peaks/troughs
from each shore-perpendicular transect
# Date Last Modified: 20250101
# To do: Update to use gpkg files and include  tile name / shoreline segments
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely import Point
import statsmodels.api as sm
import sys
import time

np.seterr(divide = 'ignore', invalid = 'ignore')

# %% Functions

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

def extract_shoreline_points_wrap(data_path, filename, shorelines_dir,
                                  shorelines_mean_dir, peaks_dir, troughs_dir, colnames,
                                  params):
    """
    Delineates for a single tile the shorelines based on water level at time of data acquisition,
    shorelines based on mean lake elevation, bar peaks, and bar troughs. Writes
    these points to *.gpkg files.

    Parameters
    ----------
    data_path : STRING
        Directory containing the *.txt files containing the transect data.
    filename : STRING
        File name of the *.txt file that contains the transect data.
    shorelines_dir : STRING
        String defining the path where shoreline points should be saved.
    shorelines_mean_dir : STRING
        String defining the path where shoreline mean points should be saved..
    peaks_dir : STRING
        String defining the path where peak points should be saved..
    troughs_dir : STRING
        String defining the path where trough points should be saved..
    colnames : LIST
        List describing the columns for the peaks and troughs DataFrames.
    params : Dictionary
        Dictionary containing the parameters needed to delineate the shorelines,
        bar peaks, and bar troughs.

    Returns
    -------
    None.

    """
    save_name_base = filename[:-4] # Define the prefix for saving outputs
    os.chdir(data_path)  # Change to directory of point *.txt files
    transect_data = pd.read_csv(filename, header = 0)  # Read point file for a single tile
    transect_data.columns = ['PointID', 'TransectID', 'Elevation', 'Distance', 'Easting', 'Northing']  # Define column names
    # Read transect gpkg to get tile name and shoreline segment attributes
    transects_gpkg = gpd.read_file( os.path.abspath(os.path.join(data_path, '..', 'Transects_mod', str(save_name_base[:-6] + 'transects.gpkg') ) ) )
    # Pre-processing all transects in a single tile (calculate slopes and trendlines)
    shorelines_df = pd.DataFrame(columns = ['num_transect'])
    shorelines_mean_df = pd.DataFrame(columns = ['num_transect'])
    peaks_df = pd.DataFrame(columns = colnames)
    troughs_df = pd.DataFrame(columns = colnames)
    # Loop over transects in data file
    for num_transect in range( min( transect_data['TransectID'] ), max( transect_data['TransectID'] ) + 1 ):
        # Debugging
        # num_transect = 878
        # print(num_transect)
        # Loop over transects using TransectID to filter
        shorelines_df, shorelines_mean_df, peaks_df, troughs_df = extract_shoreline_and_bar_points(transect_data, transects_gpkg, num_transect, shorelines_df, shorelines_mean_df, peaks_df, troughs_df, colnames, params)
    shorelines_gdf = gpd.GeoDataFrame( data = shorelines_df, geometry = shorelines_df['geometry'], crs = params['crs'] )
    shorelines_mean_gdf = gpd.GeoDataFrame( data = shorelines_mean_df, geometry = shorelines_mean_df['geometry'], crs = params['crs'] )
    peaks_gdf = gpd.GeoDataFrame( data = peaks_df, geometry = peaks_df['geometry'], crs = params['crs'] )
    troughs_gdf = gpd.GeoDataFrame( data = troughs_df, geometry = troughs_df['geometry'], crs = params['crs'] )
    shorelines_gdf.to_file(os.path.join(shorelines_dir, str ( f'{save_name_base}' + r'_shorelines.gpkg') ) )
    shorelines_mean_gdf.to_file(os.path.join(shorelines_mean_dir, str ( f'{save_name_base}' + r'_shorelines_mean.gpkg') ) )
    peaks_gdf.to_file(os.path.join(peaks_dir, str ( f'{save_name_base}' + r'_peaks.gpkg') ) )
    troughs_gdf.to_file(os.path.join(troughs_dir, str ( f'{save_name_base}' + r'_troughs.gpkg') ) )
        
        
def extract_shoreline_and_bar_points(transect_data, transects_gpkg, num_transect, shorelines_df, shorelines_mean_df, peaks_df, troughs_df, colnames, params):
    """
    Extracts the shoreline locations (mean water level and 'current' water level)
    and the location of nearshore peaks and troughs that meet specified parameters

    Parameters
    ----------
    transect_data : Pandas DataFrame
        Pandas DataFrame containing all of the transect data for a tile.
    transects_gpkg : GeoPandas GeoDataFrame
        GeoPandas GeoDataFrame containing the linestrings for the transect data
        along with tile name and shoreline segments attributes
    num_transect : INTEGER
        Unique number of the transect.
    shorelines_df : Pandas DataFrame
        DataFrame to hold shoreline points.
    shorelines_mean_df : Pandas DataFrame
        DataFrame to hold shoreline mean elevation points.
    peaks_df : Pandas DataFrame
        DataFrame to hold peaks points and properties.
    troughs_df : Pandas DataFrame
        DataFrame to hold trough points and properties.
    colnames : List
        List to hold the column names for peak and trough temporary dataframes
    params : Dictionary
        Dictionary containing the parameters for shoreline, peak, and trough delineation.

    Returns
    -------
    peaks_df : Pandas DataFrame
        DataFrame to hold peaks points and properties.
    troughs_df : Pandas DataFrame
        DataFrame to hold trough points and properties.

    """
    start_transect = time.time() # Definte the start time for processing an individual transect
    # print('Processing ', num_transect, ' out of ', max( transect_data['TransectID'] ), ' transects.') # Print the start time
    transect = transect_data[ transect_data['TransectID'] == num_transect]  # subset points from single transect
    transect_gpkg_sing = transects_gpkg.iloc[num_transect]
    row_count = transect.shape[0]  # count of points in transect
    if row_count > 0:  # process transect only if there are data
        transect = transect.sort_values(['Distance'])  # sort values by distance from lakeward end of transect
        transect = transect.reset_index(drop = True)  # Reset the index for the new dataframe containing a single transect
        transect_raw = transect.copy() # Create a copy of the raw transect data
        # Debug plotting
        # import matplotlib.pyplot as plt
        # %matplotlib qt5
        # fig, ax = plt.subplots(1)
        # ax.plot(transect.Distance, transect.Elevation)
        # ax.plot(transect_clip.Distance, transect_clip.Elev_smooth)
        # Fill data gaps:
        transect.loc[ transect['Elevation'] < -50, ['Elevation'] ] = np.nan # Set nan values
        frac_nan = transect['Elevation'].isna().sum() / len(transect)
        # Interpolate across NANs
        transect = transect.interpolate()
        transect = transect.ffill()
        transect = transect.bfill()
        # Reorient the transect if the maximum elevation is in the first 100 points (which should be lakeward end)
        if transect['Elevation'].argmax(axis = 0) < 100:
            transect['Distance'] = transect['Distance'].values[::-1]
            transect = transect.sort_values(['Distance'])  # sort values by distance from lakeward end of transect
            transect = transect.reset_index(drop = True)  # Reset the index for the new dataframe containing a single transect
        # Clip the transect from lakeward end to maximum elevation and create smoothed transect
        transect_clip = transect.iloc[0 : transect['Elevation'].argmax(axis = 0)]
        transect_clip.loc[ :, 'Elev_smooth' ] = gaussian_filter1d( transect_clip['Elevation'], sigma = 3)
        shoreline = transect_clip['Elev_smooth'].loc[ transect_clip['Elev_smooth'] > params["shoreline_elevation"] ]
        if len( shoreline) > 0:
            # Create array of elevation differences to identify interpolated flat sections
            shoreline_dif = [np.float64(0.0)]
            shoreline_dif.extend( [shoreline.iloc[i + 1] - shoreline.iloc[i] for i in range(0, len(shoreline) - 1)] )
            shoreline_dif = np.array(shoreline_dif)
            landward_troughs = shoreline.index.to_series().diff() > 1 # Find trough indices
            landward_troughs = landward_troughs[landward_troughs == True]
            # landward_troughs = landward_troughs[::-1]
            if len(landward_troughs) > 0:
                last_trough = shoreline.index[ np.argmax(landward_troughs.to_numpy() ) ]
                if ( True in (shoreline.loc[ :last_trough ] > params['trough_elevation_threshold'] ).values ):
                    last_trough = 0
                # trough_bad_filt = []
                # for count in range(0, len( landward_troughs ) ) :
                #     if ( True not in (shoreline.loc[ :landward_troughs.iloc[[count]].index.values[0] ] > params['trough_elevation_threshold'] ).values ):
                #         trough_bad_filt.append(False)
                #     else:
                #         trough_bad_filt.append(True)
                # last_trough_good = landward_troughs[ trough_bad_filt == False]        
                # last_trough = shoreline.index[ np.argmax(landward_troughs.to_numpy() ) ]
                # if True in (shoreline.loc[:last_trough] > params['trough_elevation_threshold'] ).values:
                #     last_trough = 0
            else:
                last_trough = 0
            try:
                shoreline_idx = shoreline.loc[ ( shoreline_dif != 0.0 ) & ( shoreline.index > last_trough) ].index[0]
                nearshore_nan = transect_raw.iloc[:shoreline_idx]['Elevation'].isna().sum()
                nan_flag = True if (transect_raw.iloc[shoreline_idx]['Elevation'] < -50) else False
                shoreline_point = Point( transect.iloc[shoreline_idx]['Easting'], transect.iloc[shoreline_idx]['Northing'] )
                shoreline_df_temp = pd.DataFrame(columns = ['num_transect'])
                shoreline_df_temp['num_transect'] = [num_transect]
                shoreline_df_temp['geometry'] = [shoreline_point]
                shoreline_df_temp['tile_name'] = transect_gpkg_sing['tile_name']
                shoreline_df_temp['shoreline_segment'] = transect_gpkg_sing['shoreline_segment']
                shoreline_df_temp['distance_from_lake_meters'] = transect_gpkg_sing.geometry.project(shoreline_point)
                shoreline_df_temp['nearshore_nan'] = nearshore_nan
                shoreline_df_temp['nan_flag'] = nan_flag
                shorelines_df = pd.concat( [shorelines_df, shoreline_df_temp] )
            except IndexError:
                pass
        shoreline_mean = transect_clip['Elev_smooth'].loc[ transect_clip['Elev_smooth'] > params["shoreline_elevation_mean"] ]
        if len( shoreline_mean ) > 0:
            shoreline_dif = [0.0]
            shoreline_dif.extend( [shoreline_mean.iloc[i + 1] - shoreline_mean.iloc[i] for i in range(0, len(shoreline_mean) - 1)] )
            shoreline_dif = np.array(shoreline_dif)
            landward_troughs = shoreline_mean.index.to_series().diff() > 5
            if True in landward_troughs.values:
                last_trough = shoreline_mean.index[ np.argmax(landward_troughs.to_numpy() ) ]
                if ( True in (shoreline_mean.loc[ :last_trough ] > params['trough_elevation_threshold'] ).values ):
                    last_trough = 0
                # trough_bad_filt = []
                # for count in range(0, len( landward_troughs ) ) :
                #     if ( True not in (shoreline.loc[ :landward_troughs.iloc[[count]].index.values[0] ] > params['trough_elevation_threshold'] ).values ):
                #         trough_bad_filt.append(False)
                #     else:
                #         trough_bad_filt.append(True)
                # last_trough_good = landward_troughs[ trough_bad_filt == False]        
                # last_trough = shoreline.index[ np.argmax(landward_troughs.to_numpy() ) ]
                # if True in (shoreline.loc[:last_trough] > params['trough_elevation_threshold'] ).values:
                #     last_trough = 0
                else:
                    last_trough = 0
            else:
                last_trough = 0
            try:
                shoreline_idx = shoreline_mean.loc[ ( shoreline_dif != 0.0 ) & ( shoreline_mean.index > last_trough) ].index[0]
                shoreline_mean_point = Point( transect.iloc[shoreline_idx]['Easting'], transect.iloc[shoreline_idx]['Northing'] )
                nan_flag = True if transect_raw.iloc[shoreline_idx]['Elevation'] < -50 else False
                shoreline_mean_df_temp = pd.DataFrame(columns = ['num_transect'])
                shoreline_mean_df_temp['num_transect'] = [num_transect]
                shoreline_mean_df_temp['geometry'] = [shoreline_mean_point]
                shoreline_mean_df_temp['tile_name'] = transect_gpkg_sing['tile_name']
                shoreline_mean_df_temp['shoreline_segment'] = transect_gpkg_sing['shoreline_segment']
                shoreline_mean_df_temp['distance_from_lake_meters'] = transect_gpkg_sing.geometry.project(shoreline_mean_point)
                shoreline_mean_df_temp['nan_flag'] = nan_flag
                shorelines_mean_df = pd.concat( [shorelines_mean_df, shoreline_mean_df_temp] )
            except IndexError:
                pass
        peaks, peak_props = find_peaks(transect_clip['Elev_smooth'], height = [ params["peak_lower_height"], params["peak_upper_height" ]], prominence = params["peak_prominence"])
        peak_points = [ Point( transect_clip.iloc[i]['Easting'], transect_clip.iloc[i]['Northing'] ) for i in list(peaks) ]
        peak_df_temp = pd.DataFrame(columns = colnames)
        peak_df_temp['geometry'] = peak_points
        peak_df_temp['num_transect'] = num_transect
        peak_df_temp['peak_heights'] = peak_props['peak_heights']
        peak_df_temp['prominences'] = peak_props['prominences']
        peak_df_temp['tile_name'] = transect_gpkg_sing['tile_name']
        peak_df_temp['shoreline_segment'] = transect_gpkg_sing['shoreline_segment']
        peak_df_temp['distance_from_lake_meters'] = [transect_gpkg_sing.geometry.project(point) for point in peak_points]
        peak_df_temp = peak_df_temp.sort_values( by = 'distance_from_lake_meters', ascending = False )
        peak_df_temp['peak_number_from_shore'] = peak_df_temp.index
        troughs, trough_props = find_peaks( -transect_clip['Elev_smooth'], height = [-1 * params["peak_upper_height"], -1 * params["peak_lower_height"] ], prominence = params["peak_prominence"] )
        trough_points = [ Point( transect_clip.iloc[i]['Easting'], transect_clip.iloc[i]['Northing'] ) for i in list(troughs) ]
        trough_df_temp = pd.DataFrame( columns = colnames) 
        trough_df_temp['geometry'] = trough_points
        trough_df_temp['num_transect'] = num_transect
        trough_df_temp['peak_heights'] = trough_props['peak_heights']
        trough_df_temp['prominences'] = trough_props['prominences']
        trough_df_temp['tile_name'] = transect_gpkg_sing['tile_name']
        trough_df_temp['shoreline_segment'] = transect_gpkg_sing['shoreline_segment']
        trough_df_temp['distance_from_lake_meters'] = [transect_gpkg_sing.geometry.project(point) for point in trough_points]
        trough_df_temp = trough_df_temp.sort_values( by = 'distance_from_lake_meters', ascending = False )
        trough_df_temp['trough_number_from_shore'] = trough_df_temp.index
        peaks_df = pd.concat( [peaks_df, peak_df_temp] )
        troughs_df = pd.concat( [troughs_df, trough_df_temp] )
        # print("Transect ", num_transect, ' processed in ', time.time() - start_transect, ' seconds.')
        return shorelines_df, shorelines_mean_df, peaks_df, troughs_df