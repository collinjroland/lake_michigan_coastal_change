# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:15:06 2024

@author: colli
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

np.seterr(divide='ignore', invalid='ignore')

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

    json_object = json.dumps(params, indent = 4)

    with open(loggerfilename, "w") as f:
        f.write(json_object)


def calculate_angle(point1, point2, transect):
    """
    Calculates the angle from point1 to point2.

    Parameters
    ----------
    point1 : INTEGER
        Index of the first point in the transect dataframe.
    point2 : INTEGER
        Index of the second point in the transect dataframe.
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    Returns
    -------
    angle : FLOAT
        Angle from point1 to point2.

    """
    angle = math.degrees( math.atan( ( transect.Elevation[point1] - transect.Elevation[point2] ) / ( transect.Distance[point1] - transect.Distance[point2] ) ) )
    if angle < 0: # Set negative angles to 0
        angle = 0
    return angle


def calculate_seaward_slopes(local_point, transect, params):
    """
    Calculates the average seaward slope for a single point along the transect
    using an avergae of angles calculated using the local scale window and updates
    this value in the transect DataFrame.

    Parameters
    ----------
    local_point : INTEGER
        Index of the point at which the seaward slope is being calculated.
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    params : Dictionary
        Dictionary containing the local scale window parameter (integer).

    Returns
    -------
    angle : FLOAT
        Float represeting the mean seaward slope across the local scale window whose landward end is local point.

    """
    local_window_array = np.linspace(1, params["local_scale"], params["local_scale"], dtype = int)
    local_point_2_array = local_point - local_window_array
    angle_array = [ calculate_angle(local_point, local_point_2, transect) for local_point_2 in local_point_2_array if transect.Elevation[local_point] != transect.Elevation[local_point_2] ]
    angle = np.mean(angle_array)
    return(angle)


def calculate_landward_slopes(local_point, transect, params):
    """
    Calculates the average landward slope for a single point along the transect
    using an avergae of angles calculated using the local scale window and updates
    this value in the transect DataFrame.

    Parameters
    ----------
    local_point : INTEGER
        Index of the point at which the seaward slope is being calculated.
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    params : Dictionary
        Dictionary containing the local scale window parameter (integer).

    Returns
    -------
    angle : FLOAT
        Float represeting the mean landward slope across the local scale window whose seaward end is local point.

    """
    local_window_array = np.linspace(1, params["local_scale"], params["local_scale"], dtype = int)
    local_point_2_array = local_point + local_window_array
    angle_array = [ calculate_angle(local_point_2, local_point, transect) for local_point_2 in local_point_2_array if transect.Elevation[local_point] != transect.Elevation[local_point_2] ]
    angle = np.mean(angle_array)
    return(angle)


def calculate_trendline(local_point, transect):
    """
    Calculates the trendline elevation at a local point along the transect.

    Parameters
    ----------
    local_point : INTEGER
        Index of the point along the transect.
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    Returns
    -------
    trendline : FLOAT
        Float defining the trendline elevation at the local point along the transect.

    """
    trendline = ( ( ( ( transect.loc[local_point, 'Distance'] - transect.loc[0, 'Distance'] ) *  # Numerator is the local point distance from the origin times the total elevation difference
                                              ( transect.iloc[ -1, transect.columns.get_loc( 'Elevation' ) ]  - transect.loc[0, 'Elevation' ] ) ) /
                                              ( transect.iloc[ -1, transect.columns.get_loc( 'Distance' ) ] - transect.loc[ 0, 'Distance' ] ) ) # Divide the numerator by the total distance to get fractional elevation gain
                                              + transect.loc[0, 'Elevation'] ) # Add the base elevation
    return trendline


def calculate_seaward_slope_wrap(transect, row_count, params):
    """
    Calculates the seaward slope across the transect

    Parameters
    ----------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    row_count : INTEGER
        Number of points in the modified transect.
    params : Dictionary
        Dictionary containing the local scale window parameter (integer).

    Returns
    -------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    """
    angle_array = [ calculate_seaward_slopes(local_point, transect, params) for local_point in range( params["local_scale"], row_count - params["local_scale"] - 1 ) ]
    transect.loc[params["local_scale"] : ( row_count - params["local_scale"] - 2 ) , 'SeaSlope'] = angle_array
    return transect


def calculate_landward_slope_wrap(transect, row_count, params):
    """
    Calculates the landward slope across the transect

    Parameters
    ----------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    row_count : INTEGER
        Number of points in the modified transect.
    params : Dictionary
        Dictionary containing the local scale window parameter (integer).

    Returns
    -------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    """
    angle_array = [ calculate_landward_slopes(local_point, transect, params) for local_point in range( params["local_scale"], row_count - params["local_scale"] - 1 ) ]
    transect.loc[params["local_scale"] : ( row_count - params["local_scale"] - 2 ) , 'LandSlope'] = angle_array
    return transect


def calculate_trendline_wrap(transect, row_count):
    """
    Wrapper to populate the inner portion of the first trendline elevation array in the
    transect DataFrame

    Parameters
    ----------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    row_count : INTEGER
        Number of points in the modified transect.

    Returns
    -------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    """
    transect.loc[1:row_count, 'Trendline1'] = [calculate_trendline( local_point, transect ) for local_point in range( 1, row_count ) ]
    return transect


def calculate_trendline2_wrap(transect, row_count):
    """
    Wrapper to populate the inner portion of the second trendline elevation array in the
    transect DataFrame

    Parameters
    ----------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    row_count : INTEGER
        Number of points in the modified transect.

    Returns
    -------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    """
    transect.loc[1:row_count, 'Trendline2'] = [calculate_trendline( local_point, transect ) for local_point in range( 1, row_count ) ]
    return transect
   
 
def calculate_trendline3_wrap(transect, row_count):
    """
    Wrapper to populate the inner portion of the second trendline elevation array in the
    transect DataFrame

    Parameters
    ----------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.
    row_count : INTEGER
        Number of points in the modified transect.

    Returns
    -------
    transect : Pandas DataFrame
        Pandas DataFrame containing the data for a single transect.

    """
    transect.loc[1:row_count, 'Trendline3'] = [calculate_trendline( local_point, transect ) for local_point in range( 1, row_count ) ]
    return transect

    
def preprocess_transect(clip, lakeward_limit, landward_limit, transect_data, num_transect, transect_data_mod, params):
    """
    Clips transect to highest point, calculates landward and seaward slopes
    along all transect points, and calculates the first linear trendline between
    the lowest and highest transect points and the distance between the elevation
    at every point along the transect and this trendline.

    Parameters
    ----------
    clip : BOOLEAN
        T/F whether the transect needs to be clipped to a smaller interval
    lakeward_limit : INTEGER
        Index defining the start of the transect interval to keep.
    landward_limit : INTEGER
        Index defining the end of the transect interval to keep.
    transect_data : DataFrame
        DataFrame that contains all of the transect data for a single tile.
    num_transect : Integer
        DESCRIPTION.
    transect_data_mod : DataFrame
        Empty DataFrame where modified transect data will be stored.
    params : Dictionary
        Dictionary containing various parameters for cliff feature delineation.

    Returns
    -------
    transect_data_mod : DataFrame
        DataFrame containing modified transect data.

    """
    start_transect = time.time()
    # print('Processing ', num_transect, ' out of ', max( transect_data['TransectID'] ), ' transects.')
    transect = transect_data[ transect_data['TransectID'] == num_transect]  #  Get the points from a single transect
    row_count = transect.shape[0]  # Calculate number of points in the transect
    if row_count > 0:  # process transect only if there are data
        transect = transect.sort_values(['Distance'])  # sort values by distance from seaward end of transect
        if clip == True:
            transect = transect.iloc[lakeward_limit:landward_limit]
        transect = transect.reset_index(drop=True)  # reindex
        row_count = transect.shape[0]  # Calculate number of points in the transect
        # Fill data gaps:
        transect.loc[ transect['Elevation'] < -50, ['Elevation'] ] = np.nan
        transect = transect.interpolate()
        transect = transect.ffill()
        transect = transect.bfill()
        # Create transect attributes for local slope calculations:
        zeros = np.zeros(row_count + 1)
        transect['SeaSlope'] = pd.Series(zeros)  # seaward slope (average slope between the point and nVert consecutive seaward points)
        transect['LandSlope'] = pd.Series(zeros)  # landward slope (average slope between the point and nVert consecutive landward points)
        transect['Trendline1'] = pd.Series(zeros)  # trendline #1
        transect['Difference1'] = pd.Series(zeros)  # elevations - trendline #1
        transect = transect.fillna(0)
        # Calculate slopes iterating across local scales window
        start_local = time.time()  # set start time for local scale processing
        transect = calculate_landward_slope_wrap(transect, row_count, params)
        transect = calculate_seaward_slope_wrap(transect, row_count, params)
        # print("Local scale processing took", time.time() - start_local, ' seconds.')
        # Limit the transect landwards to the highest point + local_scale:
        ind_max = np.argmax( transect['Elevation'], axis = 0) # Find the highest point of the transect
        if row_count > ind_max + params['local_scale']: # Enter if the transect is longer than the location of the highest point plus the local scale window width
            all_drop = row_count - (ind_max + params['local_scale'] + 1) # Define the interval to drop
            transect.drop( transect.tail( all_drop ).index, inplace = True) # Drop the interval
        row_count = transect.shape[0]  # Calculate the updated transect length
        transect = transect.reset_index(drop = True) # Reset the index now that the length of the DataFrame may have changed
        # Draw trendline #1 (straight line between the seaward and landward transect ends):
        transect.loc[0, 'Trendline1'] = transect.loc[0, 'Elevation']  # Set the seaward trendline elevation
        transect.iloc[ -1, transect.columns.get_loc('Trendline1') ] = transect.iloc[ -1, transect.columns.get_loc('Elevation') ]  # Set the landward trendline elevation
        transect = calculate_trendline_wrap(transect, row_count) # Calculate the inner trendline elevations
        transect['Difference1'] = transect['Elevation'] - transect['Trendline1'] # Calculate the vertical distance between the local elevation and the first trendline
        transect_data_mod = pd.concat( [transect_data_mod, transect ])  # Append modified transect to table of transects
        print("Transect ", num_transect, ' processed in ', time.time() - start_transect, ' seconds.')
        return transect_data_mod


def identify_potential_base_points(potential_base, modelled_base):
    """
    Identifys the transects that have a valid potential base point and finds
    the optimal base point for each transect (that with the largest vertical difference to the 
    first linear trendline).

    Parameters
    ----------
    potential_base : DataFrame
        DataFrame of all transect_data_mod points that meet the criteria for a potential base point.
    modelled_base : DataFrame
        DataFrame of the potential base points with the largest vertical distance to the first trendline.

    Returns
    -------
    modelled_base : DataFrame
        DataFrame of the potential base points with the largest vertical distance to the first trendline.
    cliffed_profiles : List (Series?)
        List of TransectID's that have a valid base point.
        

    """
    if potential_base.shape[0] > 0:  # Only identify base points if potential points exist
        cliffed_profiles = potential_base['TransectID'].unique()  # Identify transect ID's that contain potential base points
        for n in range(potential_base['TransectID'].min(), potential_base['TransectID'].max() + 1): # Loop across all transects between the first with a valid base point and the last with a valid base point
            for m in range(cliffed_profiles.shape[0]): # Loop across cliffed profiles with valid base points
                if n == cliffed_profiles[m]: # Enter loop if the transect actually has a valid base point
                    # Debugging
                    # n = 10
                    sub = potential_base[potential_base['TransectID'] == n] # Get all valid base points for a single transect
                    sub = sub.sort_values(by=['Difference1']) # Sort the base points by their difference from the first trendline
                    modelled_base = pd.concat( [modelled_base, sub.iloc[ [0] ] ] ) # Send the base point with the largest difference to a data frame
    return modelled_base, cliffed_profiles


def identify_potential_top_points(transect_data_mod, transect_id, modelled_base, modelled_top, cliffed_profiles, params):
    """
    Identifys potential top points for transects with a valid potential base point and finds
    the optimal top point for each transect (that with the largest vertical difference to the 
    first 2nd/3rd trendline).


    Parameters
    ----------
    transect_data_mod : DataFrame
        DataFrame containing the preprocessed transect data.
    transect_id : INTEGER
        Integer of a transect ID with a valid base point.
    modelled_base : DataFrame
        DataFrame containing modeled base point information.
    modelled_top : DataFrame
        DataFrame containing modeled top point information.
    cliffed_profiles : List (Series?)
        List of TransectID's that have a valid base point.
    params : Dictionary
        Dictionary containing various parameters for cliff feature delineation.

    Returns
    -------
    modelled_top : DataFrame
        DataFrame containing modelled top point information.

    """
    for m in range( cliffed_profiles.shape[0]) : # Loop across array with length of number of transect ID's that contain potential base points
        if transect_id == cliffed_profiles[m]: # Enter loop if the given transect ID is that of a transect with a valid base point
            # Debugging
            # n=10
            transect = transect_data_mod[ transect_data_mod['TransectID'] == transect_id ] # Get the transect data for the identified transect
            transect = transect.reset_index(drop=True) # Reset the index
            # Remove points seaward of the identified cliff base:
            transect_base = modelled_base[modelled_base['TransectID'] == transect_id]
            transect_base = transect_base.reset_index(drop=True)
            transect_base_dist = transect_base.Distance[0]
            transect.drop( transect[ transect['Distance'] < transect_base_dist ].index, inplace = True)
            transect = transect.reset_index(drop = True)
            # Draw trendline #2 between cliff base and landward transect end:
            row_count = transect.shape[0]
            zeros = np.zeros(row_count + 1)
            transect['Trendline2'] = pd.Series(zeros)  # Create Series of zeros for trendline #2
            transect['Difference2'] = pd.Series(zeros)  # Create Series of zeros for vertical difference to trendline #2
            transect = transect.fillna(0) # Fill NANs with zero
            transect.loc[0, 'Trendline2'] = transect.loc[0, 'Elevation'] # Assign first real transect elevation to first value in trendline 2 series 
            transect.iloc[ -1, transect.columns.get_loc('Trendline2') ] = transect.iloc[ -1, transect.columns.get_loc('Elevation') ] # Assign last real transect elevation to last value in trendline 2 series 
            transect = calculate_trendline2_wrap(transect, row_count) # Calculate 2nd linear trendline between the identified base and the maximum elevation point
            transect['Difference2'] = transect['Elevation'] - transect['Trendline2'] # Calculate differences between transect elevation and 2nd linear trendline
            # Find potential cliff top locations:
            potential_top = transect[ ( transect['SeaSlope'] > params["top_sea_slope"] ) & ( transect['LandSlope'] < params["top_land_slope"] ) & ( transect['Difference2'] > 0 ) ]
            if potential_top.shape[0] > 0: # Enter if a potential top point was found
                potential_top = potential_top.sort_values( by = ['Difference2'] ) # Sort potential tops by vertical difference to second linear trendline 
                modelled_top0 = potential_top.iloc[ [-1] ] # From the points that satisfy the criteria, for each transect select one with the largest vertical difference between the elevation and trendline #2:   
                # Check whether the selected point is part of within-cliff flattening:
                if ( potential_top['Distance'].max() > (modelled_top0.Distance.values[0] + params['local_scale'] ) ):
                    transect_new = transect.copy() # Create copy of transect dataframe
                    transect_top_dist = potential_top.iloc[ -1, transect.columns.get_loc('Distance') ] # Get the distance of the potential top point
                    transect_new.drop( transect_new[ transect_new['Distance'] < transect_top_dist].index, inplace = True)  # Remove points seaward of the modelled potential cliff top point
                    row_count_new = transect_new.shape[0] # Get the new length of the transect dataframe
                    # Calculate linear trendline 3 - from the first potential top point to the maximum elevation point
                    zeros_new = np.zeros(row_count_new + 1) # Create list of zeros to populate new dataframe
                    transect_new['Trendline3'] = pd.Series(zeros_new)
                    transect_new['Difference3'] = pd.Series(zeros_new)
                    transect_new = transect_new.fillna(0)
                    transect_new = transect_new.reset_index(drop=True)
                    transect_new.loc[0, 'Trendline3'] = transect_new.loc[0, 'Elevation']
                    transect_new.iloc[-1, transect_new.columns.get_loc('Trendline3')] = transect_new.iloc[-1, transect_new.columns.get_loc('Elevation')]
                    transect_new = calculate_trendline3_wrap(transect_new, row_count_new) # Calculate 3rd linear trendline between the identified top point and the maximum elevation point
                    transect_new['Difference3'] = transect_new['Elevation'] - transect_new['Trendline3'] # Calculate differnece between transect elevation and 3rd linear trendline
                    # Identify alternative potential top point
                    potential_top2 = potential_top.copy()
                    potential_top2.drop( potential_top2[ potential_top2['Distance'].values < modelled_top0.Distance.values].index, inplace = True) # Get rid of potentila top points that are seaward of the first identified top points
                    row_count_new = potential_top2.shape[0] # Get the number of remaining potential top points
                    zeros_new = np.zeros(row_count_new + 1)
                    potential_top2['Difference3'] = pd.Series(zeros_new)
                    potential_top2 = potential_top2.fillna(0)
                    potential_top2 = potential_top2.reset_index(drop = True)
                    for p in range(potential_top2.shape[0]): # Loop across the remaining potential top points
                        transect_new_temp = transect_new[ transect_new['Distance'] == potential_top2.Distance[ p ] ] # Get the transect point for the individual potential top point
                        potential_top2.iloc[ p, potential_top2.columns.get_loc('Difference3')] = transect_new_temp.Difference3 # Assign the vertical distance to the 3rd trend line to the potential top points data frame
                    # Identify alternative potential top points with a greater vertical distance than the first identified potential top point
                    potential_top2 = potential_top2[ ( ( potential_top2['Difference3'].values > 0)
                                                     & (potential_top2['Difference2'].values >=
                                                        modelled_top0.Difference2.values * params['prop_convex'] )
                                                     & ( potential_top2['Distance'].values >= modelled_top0.Distance.values + params['local_scale'] ) ) ]
                    if potential_top2.shape[0] > 0: # If there is a valid alternative potential top point enter this section
                        potential_top2 = potential_top2.sort_values( by = ['Difference2'] ) # Get the alternative top point with the greatest vertical distance
                        potential_top2.drop( ['Difference3'], axis = 1)
                        modelled_top0 = potential_top2.iloc[ [ -1 ] ]
                modelled_top = pd.concat( [modelled_top, modelled_top0] )  
    return modelled_top


def fix_outlier_top_points(transect_data_mod, modelled_base, modelled_top, row_count, params):
    if (row_count >= params["smooth_window"]): # Enter if there are enough modelled top points to exceed the smoothing window
        model = sm.OLS(modelled_top['Distance'], modelled_top['SmoothedDistance'])
        results = model.fit()
        influence = results.get_influence()
        modelled_top['StandResidual'] = influence.resid_studentized_internal
        modelled_top.loc[abs(modelled_top['StandResidual']) > 2, ['Outlier']] = 1
        fix = modelled_top[modelled_top['Outlier'] == 1]
        # 2. Delete or replace outliers with more suitable potential cliff tops:
        # (Repeat cliff top detection for the transects with outliers.)
        if fix.shape[0] > 0:
            fix = fix.reset_index(drop=True)
            modelled_top.drop(['StandResidual', 'Outlier'], axis=1)
            for c in range(fix.shape[0]):
                transect = transect_data_mod[ transect_data_mod['TransectID'] == fix.TransectID[c] ]
                transect = transect.reset_index(drop = True)
                outlier = modelled_top[ modelled_top['TransectID'] == fix.TransectID[c] ]
                # Remove points seawards from the cliff base:
                transect_base = modelled_base[modelled_base['TransectID'] == fix.TransectID[c] ]
                transect_base = transect_base.reset_index(drop = True)
                transect_base_dist = transect_base.Distance[0]
                transect.drop( transect[ transect['Distance'] < transect_base_dist].index, inplace = True)
                transect = transect.reset_index(drop = True)
                # Draw trendline #2 between cliff base and landward transect end:
                row_count = transect.shape[0]
                zeros = np.zeros(row_count + 1)
                transect['Trendline2'] = pd.Series(zeros)  # trendline #2
                transect['Difference2'] = pd.Series(zeros)  # elevation - trendline #2
                transect = transect.fillna(0)

                transect.loc[0, 'Trendline2'] = transect.loc[0, 'Elevation']
                transect.iloc[-1, transect.columns.get_loc('Trendline2')] = transect.iloc[-1, transect.columns.get_loc('Elevation')]
                transect = calculate_trendline2_wrap(transect, row_count) # Calculate 2nd linear trendline between the identified base and the maximum elevation point
                transect['Difference2'] = transect['Elevation'] - transect['Trendline2']
                # Find potential cliff top locations:
                potential_top = transect[ ( transect['SeaSlope'] > params["top_sea_slope"]) & (transect['LandSlope'] < params["top_land_slope"]) & (transect['Difference2'] > 0)]
                row_count = potential_top.shape[0]     
                zeros = np.zeros(row_count + 1)
                potential_top['SmoothedDistance'] = pd.Series(zeros)  # smoothed distance
                potential_top['DistanceFromSmoothed'] = pd.Series(zeros)  # distance from smoothed distance

                potential_top = potential_top.fillna(0)
                potential_top['SmoothedDistance'] = fix.SmoothedDistance[c]
                potential_top['DistanceFromSmoothed'] = abs(potential_top['Distance'] - potential_top['SmoothedDistance'])
                potential_top = potential_top.sort_values( by = ['DistanceFromSmoothed'] )
                potential_top = potential_top.iloc[0]

                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'PointID'] = potential_top['PointID']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Elevation'] = potential_top['Elevation']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Distance'] = potential_top['Distance']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'SeaSlope'] = potential_top['SeaSlope']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'LandSlope'] = potential_top['LandSlope']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Trendline1'] = potential_top['Trendline1']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Difference1'] = potential_top['Difference1']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Trendline2'] = potential_top['Trendline2']
                modelled_top.loc[(modelled_top['TransectID'] == potential_top['TransectID']), 'Difference2'] = potential_top['Difference2']

            row_count = modelled_top.shape[0]
            zeros = np.zeros(row_count + 1)
            modelled_top['StandResidual'] = pd.Series(zeros)  # standardized residuals
            modelled_top['Outlier'] = pd.Series(zeros)  # outliers
            modelled_top = modelled_top.fillna(0)

            model = sm.OLS(modelled_top['Distance'], modelled_top['SmoothedDistance'])
            results = model.fit()
            influence = results.get_influence()
            modelled_top['StandResidual'] = influence.resid_studentized_internal
            modelled_top.loc[abs(modelled_top['StandResidual']) > 2, ['Outlier']] = 1

            modelled_top.drop(modelled_top[modelled_top['Outlier'] == 1].index, inplace=True)  # ignore new cliff top positions if standardized residuals did not improve
    return modelled_top


def delineate_cliff_features_wrap(clip, lakeward_limit, landward_limit, data_path,
                                  filename, bluff_top_dir, bluff_base_dir,
                                  colnames, params_strict, params_lax):
    # Debugging
    # fileName = glob.glob('*.txt')[35]
    # with open(fileName, 'r') as fin:
    #     data = fin.read().splitlines(True)
    # with open(fileName, 'w') as fout:
    #     fout.writelines(data[1:])
    
    start_file = time.time()
    save_name_base = filename[:-4] # Define the prefix for saving outputs
    os.chdir(data_path)  # Change to directory of point *.txt files
    transect_data = pd.read_csv(filename, header = 0)  # Read point file for a single tile
    transect_data.columns = colnames  # Define column names
    transect_data_mod = pd.DataFrame()  # Create empty dataframe for modified transect data
    # Pre-processing all transects in a single tile (calculate slopes and trendlines)
    for num_transect in  range( min( transect_data['TransectID'] ), max( transect_data['TransectID'] ) + 1 ): # range(700, 800):
        transect_data_mod = preprocess_transect(clip, lakeward_limit, landward_limit, transect_data, num_transect, transect_data_mod, params_strict)
    transect_data_mod = transect_data_mod.reset_index(drop = True)
    # Find potential cliff base locations, elevation greater than base threshold, SeaSlope less than seaward slope upper threshold, and LandSlope greater than landward slope lower threshold:
    potential_base = transect_data_mod[ ( transect_data_mod['Elevation'] < params_strict["base_max_elev"] )
                                   & ( transect_data_mod['SeaSlope'] < params_strict["base_sea_slope"] )
                                   & ( transect_data_mod['LandSlope'] > params_strict["base_land_slope"] )
                                   & ( transect_data_mod['Difference1'] < 0 ) ]
    # From the points that satisfy the criteria, for each transect select one with the largest vertical difference between the elevation and trendline #1:
    modelled_base = pd.DataFrame(columns = potential_base.columns)
    try:
        modelled_base, cliffed_profiles = identify_potential_base_points(potential_base, modelled_base)
        # Find cliff top locations if there are transects with a cliff base point:
        if modelled_base.shape[0] > 0:
            modelled_top = pd.DataFrame() # Create empty DataFrame for cliff tops
            for transect_id in range( int( modelled_base['TransectID'].min() ), int( modelled_base['TransectID'].max() + 1 ) ): # Loop across TransectIDs with a valid base point
                modelled_top = identify_potential_top_points(transect_data_mod, transect_id, modelled_base, modelled_top, cliffed_profiles, params_strict)
            if modelled_top.shape[0] < 0: # If no modelled top points returned, use the lax parameters to identify
                for transect_id in range( int( modelled_base['TransectID'].min() ), int( modelled_base['TransectID'].max() + 1 ) ): # Loop across TransectIDs with a valid base point
                    modelled_top = identify_potential_top_points(transect_data_mod, transect_id,
                                                                 modelled_base, modelled_top, cliffed_profiles, params_lax)
            # Save the base data
            os.chdir(bluff_base_dir)
            modelled_base = modelled_base.sort_values( by = ['TransectID'] )
            modelled_base_save = modelled_base[ ['PointID', 'TransectID', 'Easting', 'Northing'] ]  # Select which columns to save; you may want to add XY coordinates if they were present
            modelled_base_save = gpd.GeoDataFrame( modelled_base_save, geometry = gpd.points_from_xy( modelled_base_save.Easting, modelled_base_save.Northing ), crs = params_strict["proj_crs"] )
            save_name_base2 = filename[:-4] + '.shp' # Define the prefix for saving outputs
            modelled_base_save.to_file(save_name_base2)  # change to header=True if exporting with header
            # Remove alongshore outliers:
            # 1. Find outliers:
            try:
                modelled_top = modelled_top.sort_values( by = ['TransectID'] )
                row_count = modelled_top.shape[0]
                zeros = np.zeros( row_count + 1 )
                modelled_top['SmoothedDistance'] = pd.Series(zeros)  # smoothed distance
                modelled_top['StandResidual'] = pd.Series(zeros)  # standardized residuals
                modelled_top['Outlier'] = pd.Series(zeros)  # outliers (https://urldefense.proofpoint.com/v2/url?u=https-3A__online.stat.psu.edu_stat462_node_172_&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=8yfrSqW1K1RIJJQehgvwMvTlPVMycUwQP0bc0m2ZrpA&m=FzFRg9yDDPqUWGYFkZANybMJTnJ5ceO8bU_NZFIOtTnG0rReObfDNmi7RpBlydEv&s=7mLUlYzS0mWlfKq6l4iPPJU4nGwCicT73n_ue03hzaA&e= ; accessed on 2021/06/04)
                modelled_top = modelled_top.fillna(0)
                modelled_top['SmoothedDistance'] = modelled_top['Distance'].rolling(window = params_strict["smooth_window"] ).median()
                modelled_top['SmoothedDistance'] = modelled_top['SmoothedDistance'].fillna(method='ffill')
                modelled_top['SmoothedDistance'] = modelled_top['SmoothedDistance'].fillna(method='bfill')
                modelled_top = fix_outlier_top_points(transect_data_mod, modelled_base, modelled_top, row_count, params_strict)
                # Save the top data:
                if modelled_top.shape[0] > 0:
                    os.chdir(bluff_top_dir)
                    save_name_top = filename[:-4] + '_top.shp'
                    modelled_top_save = modelled_top[['PointID', 'TransectID', 'Easting', 'Northing']]  # Select which columns to save; you may want to add XY coordinates if they were present
                    modelled_top_save = gpd.GeoDataFrame(modelled_top_save, geometry=gpd.points_from_xy(modelled_top_save.Easting, modelled_top_save.Northing), crs=params_strict["proj_crs"])
                    modelled_top_save.to_file(save_name_top)  # change to header=True if exporting with header
            except KeyError:
                print(r'Key Error - suspect no top points delineated')
    except UnboundLocalError:
        print(r'No cliffed profiles detected. Outputs not written for', filename)