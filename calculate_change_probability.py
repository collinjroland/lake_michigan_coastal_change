# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:59:56 2024

@author: quatlab

# Title: calculate_change_probability.py
# Author: Collin Roland
# Date Created: 20241117
# Summary: Calculates change probability rasters using Phil Wernette's 
change probability dsmchange.exe approach
# Date Last Modified: 2024112
# To do:  Nothing
"""
# %% Import packages

import glob
import numpy as np
import os
import pathlib
from pathlib import Path
import shutil
import subprocess
from subprocess import Popen, PIPE
import sys
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
# %% Set directories and read input files

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

dem_2012_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2012'
dem_2012_uncert_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2012_uncert'
dem_2020_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2020'
dem_2020_uncert_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\input\dem_2020_uncert'
computation_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\computation'
output_dir = homestr + r'\DEM_differencing_with_uncertainty\lake_michigan\outputs'

[dem_2012_fps, dem_2012_fns] = read_paths(dem_2012_dir, '.hdr')
[dem_2012_uncert_fps, dem_2012_uncert_fns] = read_paths(dem_2012_uncert_dir, '.hdr')
[dem_2020_fps, dem_2020_fns] = read_paths(dem_2020_dir, '.hdr')
[dem_2020_uncert_fps, dem_2020_uncert_fns] = read_paths(dem_2020_uncert_dir, '.hdr')

# bad_tiles = [r'2012_NCMP_WI_Michigan_47_BareEarth_1mgrid_fugro_1m', r'2012_NCMP_WI_Michigan_48_BareEarth_1mGrid_1m'] # There are some bad 2012 tiles, get rid of them
# dem_2012_fps = [i for i in dem_2012_fps if all(tile.lower() not in i.lower() for tile in bad_tiles)]
# dem_2012_fns = [os.path.basename(i) for i in dem_2012_fps]
# dem_2012_uncert_fps = [i for i in dem_2012_uncert_fps if all(tile.lower() not in i.lower() for tile in bad_tiles)]
# dem_2012_uncert_fns = [os.path.basename(i) for i in dem_2012_uncert_fps]


# %% Calculate elevation change probability (and change) with multiprocessing

exe_path = [r'D:\CJR\DEM_differencing_with_uncertainty\dsmchange.exe']

import joblib
def f(count):
    # count = 54
    dem_template = dem_2020_fns[count][17:-4]
    # Create a computational directory 
    proj_dir = computation_dir +  f'\\{dem_template}'
    # Create new project
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)
    else:
        print("Project  directory exists")
    comp_dir_sub = proj_dir
    # Get files to move
    dem_2012 = glob.glob(str(dem_2012_dir + f'\\{dem_template}*'))
    dem_2020 = glob.glob(str(dem_2020_dir + f'\\*{dem_template}*'))
    uncert_template = dem_2020_fns[count][17:-11]
    dem_2012_uncert = glob.glob(str(dem_2012_uncert_dir + f'\\{uncert_template}*'))
    dem_2020_uncert = glob.glob(str(dem_2020_uncert_dir + f'\\*{uncert_template}*'))
    files_to_move = []
    files_to_move.extend(dem_2020 + dem_2020_uncert + dem_2012 + dem_2012_uncert + exe_path)
    files_to_move = [i for i in files_to_move if '.xml' not in i]
    # Move files
    for file in files_to_move:
        shutil.copy(file, comp_dir_sub)
    # Write params.ini
    dem_2012_base = os.path.splitext( os.path.basename( files_to_move[4] ) )[0]
    dem_2012_uncert_base = os.path.splitext( os.path.basename( files_to_move[6] ) )[0]
    dem_2020_base = os.path.splitext( os.path.basename( files_to_move[0] ) )[0]
    dem_2020_uncert_base = os.path.splitext( os.path.basename( files_to_move[2] ) )[0]
    output = str('2012to2020_' + os.path.splitext( os.path.basename( files_to_move[4] ) )[0].removesuffix(r'_unmask') )
    nsim = 100
    with open( os.path.join(comp_dir_sub, 'params.ini'), 'w') as dst:
        out_str = f"input1 {dem_2012_base}\nerror1 {dem_2012_uncert_base}\ninput2 {dem_2020_base}\nerror2 {dem_2020_uncert_base}\noput {output}\nnsimulations {nsim}"
        dst.write(out_str)
    # Run change probability computation 
    os.chdir(comp_dir_sub)
    process = subprocess.call( 'dsmchange.exe', stderr=PIPE, stdout=PIPE)
    # Move output files to another folder
    outputs = glob.glob( comp_dir_sub + f'\\{output}*')
    for file in outputs:
        shutil.copy(file, output_dir)
    # shutil.rmtree(comp_dir_sub)

joblib.Parallel(n_jobs  = 6)(joblib.delayed(f)(i) for i in range(0, len(dem_2012_fns) ) ) 
# %% Calculate without parallel procesinng

for count in range(0, 73):
    # count = 57
    dem_template = dem_2020_fns[count][17:-4]
    # Create a computational directory 
    proj_dir = computation_dir +  f'\\{dem_template}'
    # Create new project
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)
    else:
        print("Project  directory exists")
    comp_dir_sub = proj_dir
    # Get files to move
    dem_2012 = glob.glob(str(dem_2012_dir + f'\\{dem_template}*'))
    dem_2020 = glob.glob(str(dem_2020_dir + f'\\*{dem_template}*'))
    uncert_template = dem_2020_fns[count][17:-11]
    dem_2012_uncert = glob.glob(str(dem_2012_uncert_dir + f'\\{uncert_template}*'))
    dem_2020_uncert = glob.glob(str(dem_2020_uncert_dir + f'\\*{uncert_template}*'))
    files_to_move = []
    files_to_move.extend(dem_2020 + dem_2020_uncert + dem_2012 + dem_2012_uncert + exe_path)
    files_to_move = [i for i in files_to_move if '.xml' not in i]
    # Move files
    for file in files_to_move:
        shutil.copy(file, comp_dir_sub)
    # Write params.ini
    dem_2012_base = os.path.splitext( os.path.basename( files_to_move[4] ) )[0]
    dem_2012_uncert_base = os.path.splitext( os.path.basename( files_to_move[6] ) )[0]
    dem_2020_base = os.path.splitext( os.path.basename( files_to_move[0] ) )[0]
    dem_2020_uncert_base = os.path.splitext( os.path.basename( files_to_move[2] ) )[0]
    output = str('2012to2020_' + os.path.splitext( os.path.basename( files_to_move[4] ) )[0].removesuffix(r'_unmask') )
    nsim = 100
    with open( os.path.join(comp_dir_sub, 'params.ini'), 'w') as dst:
        out_str = f"input1 {dem_2012_base}\nerror1 {dem_2012_uncert_base}\ninput2 {dem_2020_base}\nerror2 {dem_2020_uncert_base}\noput {output}\nnsimulations {nsim}"
        dst.write(out_str)
    # Run change probability computation 
    os.chdir(comp_dir_sub)
    process = subprocess.call( 'dsmchange.exe', stderr=PIPE, stdout=PIPE)
    # Move output files to another folder
    outputs = glob.glob( comp_dir_sub + f'\\{output}*')
    for file in outputs:
        shutil.copy(file, output_dir)
    # shutil.rmtree(comp_dir_sub)