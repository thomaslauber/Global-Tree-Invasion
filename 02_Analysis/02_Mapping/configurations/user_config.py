# This file is a configuration file that needs to be changed by the user.
# It will set the input folder, input data, output folders, settings for the upload to Google Earth Engine,
# your Google Drive, your Google Buckets etc.
# Please define all the following variables in PART 1, please do not change anything
# in PART 2.
# ------------------------------------------------------------------------------------------------------------- #
# PART 1: To adapt
# ------------------------------------------------------------------------------------------------------------- #
# Overall settings: Input (data, etc.), Output directory, Settings for upload/download to GEE and the Python API
# for Google Earth Engine #
# -------------------------------------------------------------------------------------------------------------- #
# Input #
# ----- #
# Pipeline choice: Is it a global analysis or a local (observations only in a specific region)?
pipelineChoice = 'global'

# Specify the column names where the latitude and longitude information is stored
latString = 'avglat'
longString = 'avglon'

# Input the name of the classification property
classPropList = ['instat_class']

# Output #
# ------ #
# minModel, allCovs
# outputPath = '/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/output/wUpsampling'
# outputPath = '/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/output/woUpsampling'
# outputPath = '/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/output/smote'
outputPath = '/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/output/'

# Python API for Google Earth Engine setting #
# ------------------------------------------ #
# !! You must specify the full path of the executable if the executable is not scoped from root
bashFunction_EarthEngine = '/Users/Thomas/opt/anaconda3/envs/ee/bin/earthengine'

# Google Cloud #
# ------------ #
# Initialize the Google cloud bucket
# IMPORTANT: If you want to store the results in a subfolder in your bucket
#            please PRECISELY define the path to this subfolder:
#            Example: niamh_bucket_31415927/subfolder1/subsubfolder1.
bucketOfInterest = 'crowtherlab_gcsb_t3'

# Google Earth Engine User settings #
# --------------------------------- #
# Initialize your username in Google Earth Engine
usernameFolderString = 'crowtherlab/t3'

# Initialize project folder name in Google Earth Engine
# Here all of the assets will be stored in
# !! You should create this folder immediately under the home asset directory before the script is run !!
# projectFolder = 'InvasiveSpecies/InvasionMapping'
# projectFolder = 'InvasiveSpecies/InvasionMapping_noUpsampling'
projectFolder = 'InvasiveSpecies/InvasionMapping'

# Data preparation #
# ---------------- #

# Input list of the covariables to be used #
# pick 000cm all the time
covariates_reduced = [
    'WFP_DistanceToPorts',
    'GHS_Population_Density',
    'CHELSA_BIO_Annual_Mean_Temperature',
    'CHELSA_BIO_Annual_Precipitation',
    'SG_Absolute_depth_to_bedrock',
    'SG_Coarse_fragments_000cm',
    'SG_Sand_Content_000cm',
    'SG_Silt_Content_000cm',
    'SG_Soil_pH_H2O_000cm',
]

covariateList = covariates_reduced

# Data type #
# in future maybe add the choice of categorical
datatype = 'categorical'

# ONLY for categorical data #
# this needs to be in the same data type as
# the categories are in the input data
categories = [0, 1]

# Grouping variable #
groupingVariable = 'Resolve_Biome'

# Imputation strategy #
# chose either: 'remove', 'univariateMean', 'univariateMedian'
imputeStrategy = 'remove'

# Data preparation #
# ---------------- #
# Input the proportion of variance that you would like to cover
# running the extrapolation vs. interpolation function
propOfVariance = 90

# k-fold CV #
# -------------- #
# Choose the number of folds for k-fold CV
k = 10

# Input the model type; i.e., is this a classification (on categorical data) or a regression (on continuous data)?
modelType = 'PROBABILITY'

# Input the resampling type if great class unbalances are present; either 'SMOTE' or 'None'
resamplingStrategy = 'None'

# Chose the size of spatial blocks for blockCV (in km)
blockCvSize = 250

# Bootstrapping preparation #
# ------------------------- #
# Stratification variable #
stratVariable = 'Resolve_Biome'

# stratificationChoice; either 'numberOfSamples' or 'area'
# stratificationChoice = 'numberOfSamples'
stratificationChoice = 'area'

# Chose an appropriate number to sample #
# from each category in the stratification variable #
# samplesPerStratVariable = 200

# the size of each bootstrap sample
bootstrapModelSize = 10000

# number of bootstrap samples
noOfBootstrapSamples = 50

# ------------------------------------------------------------------------------------------------------------- #
# PART 2: PLEASE DO NOT CHANGE ANYTHING HERE
# ------------------------------------------------------------------------------------------------------------- #

## Import the modules of interest, the packages need to be installed on the computer!
import pandas as pd
import numpy as np
import subprocess
import time
import datetime
import ee as ee
ee.Initialize()
import math
from itertools import cycle
import os
from os import listdir
from os.path import isfile, join
import glob

# These are needed for the grid searching script
from typing import Iterable, Any
from itertools import product

# These are needed for the extrapolation vs. interpolation script and the aoa calculation
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations
from itertools import repeat

# These packages are needed for oversampling unbalanced binary classification data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random

# These are needed for the performance evaluation
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from plotnine import *
import plotnine
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

#  Settings not needed to be changed!
# ------------------------------------ #
# Overall settings: Input (data, etc.), Output directory, Settings for upload/download to GEE and the Python API
# for Google Earth Engine #
# -------------------------------------------------------------------------------------------------------------- #
# Input #
# ----- #

# Path to the composite data on Google Earth Engine
composite = 'projects/crowtherlab/t3/InvasiveSpecies/20220419_InvasionComposite_final'

# Normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 30

# Longer wait time (in seconds) for "wait and break" cells
longWaitTime = 600

# Specify the bashfunction gsutil
bashFunctionGSUtil = '/Users/Thomas/opt/anaconda3/envs/ee/bin/gsutil'

# CV strategy: either 'foldID' for random CV or 'blockCV' for block CV
# cvFoldString = 'foldID'
cvFoldString = 'blockCV'

# Specify the arguments to these functions
CRStoUse = 'EPSG:4326'
arglist_preEEUploadTable = ['upload','table']
arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString, '--csv_delimiter', ',', '--csv_qualifier', '"', '--crs', 'EPSG:4326']
arglist_preGSUtilUploadFile = ['cp']
arglist_preGSUtilUploadDirectory = ['-m', 'cp', '-r']
arglist_downloadDirectory = ['-m ', 'cp', '-r', 'dir']
formattedBucketOI = 'gs://'+bucketOfInterest
assetIDStringPrefix = '--asset_id='
arglist_CreateCollection = ['create','collection']
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_Delete = ['rm','-r']
stringsOfInterest = ['Asset does not exist or is not accessible']


# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder
bashCommandList_downloadDirectory = [bashFunctionGSUtil]+arglist_downloadDirectory

# Compute the number of bootstrap samples
seedsToUseForBootstrapping = list(range(1, noOfBootstrapSamples))

# define geometry for exportation purposes
unboundedGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)
