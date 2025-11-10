# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:57:02 2024

@author: quatlab

# Title: make_cross_section_functions.py
# Author: Collin Roland
# Date Created: 20241121
# Summary: Functions for extracting elevations to transects
# Date Last Modified: 20241231
# To do: Nothing
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


# %% Functions


def temp_merge(dem_files):
    mem = MemoryFile()
    merge(dem_files, dst_path = mem.name)
    merged_dem = rasterio.open(mem)
    return(merged_dem)

    
def densify_geometry(line_geometry, step, crs = None):
    # crs: epsg code of a coordinate reference system you want your line to be georeferenced with
    # step: add a vertice every step in whatever unit your coordinate reference system use.
    length_m = line_geometry.length # get the length
    xy=[] # to store new tuples of coordinates
    for distance_along_old_line in np.arange(0,int(length_m),step): 
        point = line_geometry.interpolate(distance_along_old_line) # interpolate a point every step along the old line
        xp,yp = point.x, point.y # extract the coordinates
        xy.append((xp,yp)) # and store them in xy list
    new_line=LineString(xy) # Here, we finally create a new line with densified points.  
    if crs != None:  #  If you want to georeference your new geometry, uses crs to do the job.
        new_line_geo=gpd.geoseries.GeoSeries(new_line,crs=crs) 
        return new_line_geo
    else:
        return new_line
  


def replace_line_end(adjacent_line, line, snap_tolerance):
    '''
    adjacent_line = Shapely Linestring for line whose endpoint that will be snapped to line
    line = Shapely Linestring for line that is being snapped to
    snap_tolerance = float, distance in meters that snaps can occur across
    
    Snaps start/endpoint of a linestring and regenerates line
    '''
    # Debugging
    # adjacent_line = seg_1
    # line = seg_2
    # snap_tolerance = 20.0
    start_adj = shapely.Point(adjacent_line.coords.xy[0][0], adjacent_line.coords.xy[1][0])
    end_adj = shapely.Point(adjacent_line.coords.xy[0][-1], adjacent_line.coords.xy[1][-1])
    adj_near = shapely.ops.nearest_points(line, adjacent_line.boundary)[1]
    adj_near_snap = shapely.snap(adj_near, line, tolerance = snap_tolerance)
    distance = adj_near.distance([start_adj, end_adj])
    adj_line_points = [shapely.Point(i) for i in adjacent_line.coords]
    if adj_near.distance(start_adj) < adj_near.distance(end_adj):
        adj_line_points[0] = adj_near_snap
    else:
        adj_line_points[-1] = adj_near_snap
    adjacent_line = shapely.LineString(adj_line_points)
    return adjacent_line


def prep_shoreline_and_tile(shoreline_path, tiles_path, count):
    """
    Prepares the inputs for the function gen_xsec_wrap. Gets a single tile of interest,
    clips it to not overlap previously processed tiles, and generates a merged DEM that
    covers a buffered version of the tile.

    Parameters
    ----------
    shoreline_path : string
        Path to the shoreline file (.shp)
    tiles : GeoDataFrame
       GeoDataFrame of the DEM tile index

    Returns
    -------
    shoreline_clip : GeoDataFrame
        GeoDataFrame of the shoreline that is clipped to the cleaned DEM tile
    tile_sing_clean : GeoDataFrame
        GeoDataFrame of the cleaned DEM tile that does not overlap previous tiles
    tile_sing : GeoDataFrame
        GeoDataFrame of the DEM tile that does overlap previous tiles
    """
    tiles = gpd.read_file(tiles_path)
    shoreline = gpd.read_file(shoreline_path)
    shoreline = shoreline.to_crs(tiles.crs)
    tile_sing = tiles.iloc[ [ count ] ]  # pull out a single tile
    tile_name = os.path.splitext(tile_sing.location.iloc[0])[0]
    tiles_prior = tiles.iloc[ 0 : count - 1 ]
    if count > 0:
        tile_sing_clean = tile_sing.overlay(tiles_prior, how = 'difference' )  
    else:
        tile_sing_clean = tile_sing
    bounds = tile_sing_clean.geometry.bounds
    shoreline_clip = gpd.clip(shoreline.geometry, tile_sing_clean.geometry) # clip shoreline to the clipped tile
    return shoreline_clip, tile_sing_clean, tile_sing, tile_name


def make_merged_dem(count, tiles_path, tile_sing, tile_buffer, dem_dir, dem_fps):
    """
    Identified all of the adjacent tiles to tile of interest and merges all of
    the DEMs that are associated with these tiles.
    Parameters
    ----------
    tiles : GeoDataFrame
        GeoDataframe of the DEM tile index
    tile_sing : GeoDataFrame
        GeoDataFrame for the tile of interest.
    tile_buffer : INTEGER
        INTEGER DEFINING THE SINGLE SIDED BUFFER LENGTH.
    dem_dir : STRING
        String defining the path to the DEM tiles directory
    dem_fps : array of strings

    Returns
    -------
    merged_dem : RASTERIO MEMORY FILE
        RASTERIO MEMORY FILE FOR THE MERGED DEM

    """
    tiles = gpd.read_file(tiles_path)
    tile_name = os.path.splitext(tile_sing.location.iloc[0])[0]
    tile_sing_buffer = shapely.buffer(tile_sing.geometry, tile_buffer)
    tile_sing_buffer = gpd.GeoDataFrame( geometry = tile_sing_buffer, crs = tile_sing.crs )
    merge_tiles = gpd.sjoin(tiles, tile_sing_buffer)
    merge_tiles_paths = [os.path.join( dem_dir, ( os.path.splitext(i)[0] + '.tif' ) ) for i in merge_tiles['location'] ]
    merge_tiles_names = [os.path.splitext( os.path.basename(i) )[0] for i in merge_tiles_paths]
    merge_tiles = [i for i in dem_fps if any(name.casefold() in i.casefold() for name in merge_tiles_names) ]
    merge_dem_name = f'merge_{count}.tif'
    merged_dem = temp_merge(merge_tiles)
    return merged_dem


def merge_and_smooth_shorelines(shoreline_clip, simp_tol, snap_tol):
    """
    Takes in the shoreline clipped to the cleaned tile, merges nearly touching
    segments, and smooths the segments

    Parameters
    ----------
    shoreline_clip : GeoSeries
        GeoSeries containing the clipped shoreline segments
    snap_tol : INTEGER
        Integer that defines the snapping tolerance for defining touching
        segments

    Returns
    -------
    merge_shorelines : Array of Shapely objects?
        Array of Shapely objects containing the merged and smoothed shoreline segments

    """
    merge_shorelines = shapely.unary_union(shoreline_clip.geometry) # Turn individual GeoSeries of LineStrings into MultiLineString 
    if type(merge_shorelines.geom_type) == 'MultiLineString':
        merge_shorelines = shapely.ops.linemerge(merge_shorelines) # Merge intersecting segments
        merge_shorelines = gpd.GeoSeries(data = merge_shorelines, crs = shoreline_clip.crs) # Turn MultiLineString into GeoSeries after merging
    else:
        merge_shorelines = gpd.GeoSeries(data = merge_shorelines, crs = shoreline_clip.crs) # Turn LineString into GeoSeries
    if merge_shorelines.geometry[0].geom_type == 'MultiLineString': # Break back into LineStrings if is MultiLineString
        merge_shorelines = merge_shorelines.geometry.explode(ignore_index = True) # Explode the geometries
        for count in range (0, len( merge_shorelines) ): # Loop over geometries in the merged shoreline
            segment = merge_shorelines.iloc[count] # Identify a single segment
            segment_mod = merge_shorelines.iloc[count] # Make a copy variable for the single segment
            segment_distances = segment.distance( [i for i in merge_shorelines if i != segment] ) # Calculate distances to segments (?)
            segment_distances = np.insert(segment_distances, count, 1000.0) # Insert value 1000 at the count (self) position in segment distances array
            touching_segments = np.where(segment_distances < snap_tol)[0].tolist() # Identify touching segements (those with a distance less than the snap_tol paramete)
            if len(touching_segments) > 0: # Enter if there are touching segments
                for i in touching_segments: # Loop over touching segments
                    segment_mod = replace_line_end(segment_mod, merge_shorelines.iloc[i], segment_distances[i] + 2.0) # Execute function to match segment ends (?)
                merge_shorelines.iloc[count] = segment_mod # Replace the current geometry with the modified segment with matching ends
        merge_shorelines = shapely.unary_union(merge_shorelines.geometry) # Reunify geometries to merge touching linestrings that have been fixed
        merge_shorelines = shapely.ops.linemerge(merge_shorelines) # Reunify geometries to merge touching linestrings that have been fixed
        merge_shorelines = gpd.GeoSeries(data = merge_shorelines, crs = shoreline_clip.crs) # Turn fixed shorelines from Shapely objects into GeoSeries
    merge_shorelines = merge_shorelines.geometry.explode(index_parts = True) # Explode the geometries after all of the fixing operations
    for count in range(0, len( merge_shorelines[0] ) ): # Loop over individual shoreline segments
        geom = merge_shorelines[0][count]
        geom = shapelysmooth.chaikin_smooth(geom.simplify(simp_tol), iters = 7) # Smooth the segment
        merge_shorelines[0][count] = geom # Replace shoreline segment with smoothed segment
    return merge_shorelines


def calculate_bearing(shoreline_clip, tempPointList_xsec):
    """
    

    Parameters
    ----------
    shoreline_clip : GeoSeries
        GeoSeries containing the clipped shoreline segements
    tempPointList_xsec : array of Shapely Points?
        Array of Shapely Points that represent the intersection points of the transects and the shoreline segments.

    Returns
    -------
    temp_angle : array
        Array of angles representing the transect angle that is perpendicular to the local shoreline angle of the merged and smoothed shoreline.

    """
    transformer = pyproj.Transformer.from_crs(shoreline_clip.crs, 6318)
    geodesic = pyproj.Geod(ellps = 'WGS84')
    temp_angle = [] # Create empty array for bearing of cross-section points
    for count_2, value in enumerate(tempPointList_xsec): # Loop over transect intersection points
        if count_2 > 0: # Only calculate backward bearing for transect intersections that are not the first
            lat1, long1 = transformer.transform(tempPointList_xsec[count_2 - 1].x, tempPointList_xsec[count_2 - 1].y)
            lat2, long2 = transformer.transform(tempPointList_xsec[count_2].x, tempPointList_xsec[count_2].y)
            fwd_azimuth,back_azimuth,distance = geodesic.inv(long1, lat1, long2, lat2)
            orth_angle = back_azimuth - 90
            temp_angle.append(orth_angle)
        if count_2 == 0: # Only calculate forward bearing for transect intersections that are the first
            count_2 = 1
            lat1,long1 = transformer.transform(tempPointList_xsec[count_2 - 1].x, tempPointList_xsec[count_2 - 1].y)
            lat2,long2 = transformer.transform(tempPointList_xsec[count_2].x, tempPointList_xsec[count_2].y)
            fwd_azimuth,back_azimuth,distance = geodesic.inv(long1,lat1,long2,lat2)
            orth_angle = back_azimuth - 90
            temp_angle.append(orth_angle)
    return temp_angle

    
def make_xsec_points(transects, merged_dem):
    transects = transects
    all_points = pd.DataFrame()
    for count,transect in enumerate(transects.geometry):
        points = [Point(coord[0], coord[1]) for coord in transect.coords]
        x = [coord[0] for coord in transect.coords]
        y = [coord[1] for coord in transect.coords]
        ID = [count]*len(points)
        near_dist = [distance(points[0],point) for point in points]
        elevation = [x[0] for x in merged_dem.sample(transect.coords)]
        temp_df = pd.DataFrame({'ID_1':ID,'RASTERVALU':elevation,'NEAR_DIST':near_dist,'Easting':x,'Northing':y})
        all_points= pd.concat([all_points,temp_df])
    all_points = all_points.reset_index(drop=True)
    all_points['FID']= all_points.index
    all_points = all_points[['FID','ID_1','RASTERVALU','NEAR_DIST','Easting','Northing']]
    return(all_points)


def make_transect_intersection_points(shoreline_sing_3, xsec_spacing):
    """
    Creates points evenly spaced along shoreline segment from which transects 
    will originate/intersect

    Parameters
    ----------
    shoreline_sing_3 : GeoSeries
        Shoreline segment
    xsec_spacing : INTEGER
        Integer defining spacing between cross sections (transects)

    Returns
    -------
    tempPointList_xsec : list
        List of intersection points between transects and shoreline segment.
    tempLineList_xsec : list
        Empty list.
    tempStartPoints_xsec : list
        Empty list.
    tempEndPoints_xsec : list
        Empty list.

    """
    num_points = [] # Create empty array to store the number of transects along the shoreline segment
    num_points = int(shoreline_sing_3.iloc[0].length / xsec_spacing) # Calculate number of transects
    lenSpace = np.linspace(xsec_spacing, shoreline_sing_3.iloc[0].length, num_points) # Compute the distance array of the transect locations along the shoreline segment
    tempPointList_xsec = [] # Create empty array for points where transects intersect shoreline segment
    tempLineList_xsec = [] # Create empty array for lines
    tempStartPoints_xsec = [] # Create empty array for transect start points
    tempEndPoints_xsec = [] # Create empty array for transect end points
    for space in lenSpace: # Loop over the transect distances along the segment
        tempPoint_xsec = (shoreline_sing_3.interpolate(space)) # Identify the location of the transect intersection along the shoreline segment
        tempPointList_xsec.append(tempPoint_xsec) # Append transect intersection points to list
    return tempPointList_xsec, tempLineList_xsec, tempStartPoints_xsec, tempEndPoints_xsec


def gen_xsec(point, xsec_angle, poslength, neglength, samp_spacing, merged_dem, crs):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.
    
    Will plot the line on a 10 x 10 plot.
    '''
    # unpack the first point
    x = point.iloc[0].xy[0][0]
    y = point.iloc[0].xy[1][0] 
    # Calculate the transect end point
    endy = y + poslength * math.cos( math.radians( xsec_angle[0] ) )
    endx = x + poslength * math.sin( math.radians (xsec_angle[0] ) )
    end_point = Point(endx, endy)   
    # Calculate the transect start point
    starty = y + neglength * math.cos( math.radians( xsec_angle[0] + 180 ) )
    startx = x + neglength * math.sin( math.radians( xsec_angle[0] + 180 ) )
    start_point = Point(startx, starty)
    # Make a line from the points
    xsec_line = LineString( [start_point, end_point] )
    xsec_line = densify_geometry(xsec_line, samp_spacing, crs) # Densify the geometry using the designated sample spacing
    # Calculate median elevations of end and start intervals of the line to assess landward direction
    start_coords = xsec_line.geometry[0].coords[ 0 : int( neglength ) ] # Get the start interval point coordinates
    start_elevs = np.array( [sample[0] for sample in merged_dem.sample( start_coords ) ] ) # Get the start interval elevations
    start_elevs[ start_elevs < 0.0 ] = np.nan # Set start elevations less than 0 to NAN
    start_elev = np.nanmedian( start_elevs ) # Get the median start interval elevation excluding NANs
    if np.isnan(start_elev): # If the start interval is all NANs, set to 100
        start_elev = 100.0
    end_coords = xsec_line.geometry[0].coords[int(neglength):int(neglength + poslength)] # Get the end interval point coordinates
    end_elevs = np.array([sample[0] for sample in merged_dem.sample(end_coords)]) # Get the end interval elevations
    end_elevs[end_elevs < 0.0] = np.nan # Set the end interval elevations less than 0 to NAN
    end_elev = np.nanmedian(end_elevs) # Get the median end interval elevation excluding NANs
    if np.isnan(end_elev): # If the end interval is all NANs, set to 100
        end_elev = 100.0
    # Reorient line if start interval elevation greater than end interval elevation (want start in the lake)
    if start_elev >= end_elev: # If the start elevation is greater than end elevation, go about fixing the transect orientation
        xsec_angle_mod = xsec_angle[0] + 180 # Rotate the transect angle by 180 degrees
        # Calculate the modified end point
        endy2 = y + poslength * math.cos(math.radians( xsec_angle_mod) )
        endx2 = x + poslength * math.sin(math.radians( xsec_angle_mod) )
        end_point2 = Point(endx2, endy2) 
        # Calculate the modified start point
        starty2 = y + neglength * math.cos(math.radians(xsec_angle_mod + 180) )
        startx2 = x + neglength * math.sin(math.radians(xsec_angle_mod + 180) )
        start_point2 = Point(startx2, starty2)
        # Generate a line from points
        xsec_line2 = LineString( [start_point2, end_point2]) 
        xsec_line2 = densify_geometry(xsec_line2, samp_spacing, crs = crs)
        # Calculate median elevations of end and start intervals of the modified line to assess landward direction
        start_coords = xsec_line2.geometry[0].coords[0 : int( neglength )]
        start_elevs = np.array( [ sample[0] for sample in merged_dem.sample( start_coords ) ] )
        start_elevs[start_elevs < 0.0] = np.nan
        start_elev2 = np.nanmedian(start_elevs)
        if np.isnan(start_elev2):
            start_elev2 = 100.0
        end_coords = xsec_line2.geometry[0].coords[ int( neglength ) : int( neglength + poslength ) ]
        end_elevs = np.array( [ sample[0] for sample in merged_dem.sample( end_coords ) ] )
        end_elevs[ end_elevs < 0.0 ] = np.nan
        end_elev2 = np.nanmedian( end_elevs ) 
        if np.isnan( end_elev ):
            end_elev2 = 100.0
        if ( start_elev2 < end_elev2 ):
            xsec_line = xsec_line2
            start_point = start_point2
            end_point = end_point2
    return xsec_line, start_point, end_point


def gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tol, samp_spacing, snap_tol, merged_dem, crs):
    """
    

    Parameters
    ----------
    shoreline_clip : GeoSeries
        GeoSeries containing the clipped shoreline segments
    outdir : string
        String containing the path to the output directory
    tile_name : TYPE
        DESCRIPTION.
    poslength : TYPE
        DESCRIPTION.
    neglength : TYPE
        DESCRIPTION.
    xsec_spacing : TYPE
        DESCRIPTION.
    simp_tolerance : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    merged_dem : TYPE
        DESCRIPTION.
    crs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    merge_shorelines = merge_and_smooth_shorelines( shoreline_clip, simp_tol, snap_tol)
    for count_1 in range(0, len( merge_shorelines ) ): # Loop over geometry parts of the merged and smoothed shorelines
        shoreline_sing_1 = gpd.GeoSeries(data = merge_shorelines.iloc[count_1], crs = shoreline_clip.crs) # Change Shapely object into GeoSeries
        if shoreline_sing_1.geometry.geom_type[0] == 'MultiLineString': # Exploode if MultiLineString
            shoreline_sing_2 = shoreline_sing_1.geometry.explode(index_parts = True)
        else:
            shoreline_sing_2 = shoreline_sing_1
        for count_4 in range(0, len( shoreline_sing_2 ) ): # Loop over segments of individual geoemtry part
            shoreline_sing_3 = gpd.GeoSeries( data = shoreline_sing_2.iloc[count_4], crs = shoreline_clip.crs) # Change segment to GeoSeries
            if (shoreline_sing_3.geometry.length[0] >= ( 2 * xsec_spacing ) ):   # Create transect if segment is longer than twice the transect spacing
                # Create cross-section points
                tempPointList_xsec, tempLineList_xsec, tempStartPoints_xsec, tempEndPoints_xsec = make_transect_intersection_points(shoreline_sing_3, xsec_spacing)
                # Calculate bearing at cross-section points
                temp_angle = calculate_bearing(shoreline_clip, tempPointList_xsec)
                # Generate cross-section lines
                xsec_lines = [] # Create empty list for xsec lines
                xsec_starts = [] # Create empty list for xsec starts
                xsec_ends = [] # Create empty list for xsec ends
                for count_3, value in enumerate(temp_angle): # Loop over the perpendicular angle array, which has the same length as the number of transects
                    [xsec_line, start_point, end_point] = gen_xsec(tempPointList_xsec[count_3], value, poslength, neglength, samp_spacing, merged_dem, crs) # Make the cross section (transect)
                    xsec_lines.append (xsec_line.iloc[0] )
                    xsec_starts.append(start_point)
                    xsec_ends.append(end_point)  
                # Convert to GDF
                xsec_lines = gpd.GeoDataFrame(geometry = xsec_lines, crs = crs)
                xsec_starts = gpd.GeoDataFrame(geometry = xsec_starts, crs = crs)
                xsec_ends = gpd.GeoDataFrame(geometry = xsec_ends, crs = crs)
                xsec_shoreline = gpd.GeoDataFrame(geometry = shoreline_sing_3.geometry, crs = crs)  
                # Add shoreline and tile attributes
                xsec_lines['tile_name'] = tile_name
                xsec_starts['tile_name'] = tile_name
                xsec_ends['tile_name'] = tile_name
                xsec_shoreline['tile_name'] = tile_name
                xsec_lines['shoreline_segment'] = str(count_1)+'_'+str(count_4)
                xsec_starts['shoreline_segment'] = str(count_1)+'_'+str(count_4)
                xsec_ends['shoreline_segment'] = str(count_1)+'_'+str(count_4)
                xsec_shoreline['shoreline_segment'] = str(count_1)+'_'+str(count_4)
                # Write outputs
                tmp_start = os.path.join(outdir,'StartPoints',(tile_name+'_start_points_'+str(count_1)+'_'+str(count_4)+'.gpkg'))
                tmp_end = os.path.join(outdir,'EndPoints',(tile_name+'_end_points_'+str(count_1)+'_'+str(count_4)+'.gpkg'))
                tmp_lines = os.path.join(outdir,'Transects',(tile_name+'_transects_'+str(count_1)+'_'+str(count_4)+'.gpkg'))
                tmp_shore = os.path.join(outdir,'Shorelines',(tile_name+'_shoreline_'+str(count_1)+'_'+str(count_4)+'.gpkg'))
                xsec_lines.to_file(tmp_lines, driver = 'GPKG', layer = tile_name+'_transects_'+str(count_1)+'_'+str(count_4) )
                xsec_starts.to_file(tmp_start, driver = 'GPKG', layer = tile_name+'_start_points_'+str(count_1)+'_'+str(count_4))
                xsec_ends.to_file(tmp_end, driver = 'GPKG', layer = tile_name+'_end_points_'+str(count_1)+'_'+str(count_4))
                xsec_shoreline.to_file(tmp_shore, driver = 'GPKG', layer = tile_name+'_shoreline_'+str(count_1)+'_'+str(count_4))