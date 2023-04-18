# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:10:22 2021

@author: Niamh M. Robmann & Thomas Lauber

# INTRO
## Set up

### Loading configurations
The set up will load the configurations (variable names, paths, etc.),
the classifiers, functions to prepare folder settings locally, in the
Google Cloud Storage Bucket and Google Earth Engine asset's directory and
finally the functions to run the pipeline. If you haven't done so already,
now you need to check the configurations directory and adapt the
following two files:

- user_config.py
- classifiers.py.

Please follow the instructions in the files.

## Pipeline
The second part of this script will run the pipeline itself.
The pipeline step again, is subdivided into the main steps of the workflow.

### Data preparation
Input: file.csv, with the following columns [Lat, Long, VariableOfInterest]

This part of the pipeline will create a clean version of the input data
for the pipeline. This data contains your variable of interest and the
corresponding coordinate values (LAT, LON). In a first step the original
coordinates will be paired up with the corresponding pixel coordinates
(if a coordinate does not possess an exact corresponding pixel coordinate
 it will be assigned to the closest pixel coordinate). This data is then
paired with the chosen covariables (environmental data) from the composite
(and information about the ecoregion and biome). If a covariable does not
contain a value at a specific pixel coordinate, it will be assigned a value
 from the closest pixel coordinate. Missing data in the resulting matrix can
 either be removed or imputed by the 'mean' or 'median'.

### Assessment of the level of extrapolation vs. interpolation
This step will assess the level of extrapolation vs. interpolation
in your data based on the covariates space.

### The main step: Hyper-parameter selection, model training and predictions
In a first step user-defined classifiers are being evaluated using a
grid search based on k-fold cross validation. The classifier that gives
the best accuracy value is then used to train the random forest model on
the input data. With the trained model based on the input data predictions
on a the entire globe are computed.

The model training and prediction can be done using two approaches:

#### Simple map
The 'Simple Map' procedure will run the training and prediction on the
entire set of data. Additionally, you will get values for the feature
importances of your chosen covariates. This can be used to evaluate your
choice of covariates and for model improvements (repetition of the
modelling using only the top covariates).

#### Bootstrapped map
The 'Bootstrapped Map' procedure aims at giving an estimation of how
accurate your model predicts data. The procedure will produce multiple
bootstrapped samples from your input data. The model is trained on each
of the bootstrapped samples and predictions are made for each of them as well.
You will end up with multiple models (maps). Based on these maps a final
map is produced by computing the mean and standard deviation of all these
maps at each pixel coordinate. Thus, we get an estimate of
how accurate our model is.
"""


"""
---------------------------------
1) Set up
Loading configurations and necessary functions
---------------------------------
"""

import os
os.chdir('/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/code/02_Analysis/01_Global_Mapping')

# load all modules ands configs
from configurations.user_config import *
from configurations.setUp import *
from configurations.classifiers import *
from modules.pipeline import *

# set up output folders
setUp(outputPath, projectFolder)

"""
---------------------------------
2) Pipeline
---------------------------------
"""
# i) Data preparation
# Get data
inputFolder = '/Users/Thomas/Projects/SpeciesInvasion/InvasionMapping/data'
currentInputVariables = ['lat', 'avglon', 'in.stat']
phyloData_name = 'GFBI_invasion_foranalyses_allplots.csv'

phyloData = pd.read_csv(inputFolder + '/' + phyloData_name, sep=',', index_col=False)
data = phyloData[currentInputVariables]
data['instat_class'] = data['in.stat'].apply(lambda x: 0 if x=='non-invaded' else 1)
data['avglat'] = data['lat']; data = data.drop('lat', axis=1); data = data.drop('in.stat', axis=1)
print('before droping nas', data.shape)
data = data.dropna(axis=0).reset_index(drop=True)
print('Shape of input data: ', data.shape, '\n')
print('classification properties to be modeled: ', classPropList, '\n')
print('Input data:')
print(data.head())
print('Number of invaded vs non-invaded plots:', sum(data['instat_class'] == 1), 'vs.', sum(data['instat_class'] == 0), '\n')

dataPrepFun(data, classPropList, covariateList, imputeStrategy, resamplingStrategy, 'bootstrapped', 'Resolve_Biome', 'mode', True, distanceToFill = 100000, noOfDecimals = 7)
finalData = pd.read_csv(outputPath+'/outputs/dataPrep/finalCorrMatrix.csv')
print('shape of final data: ', finalData.shape)
print('final data to use: ')
print(finalData.head())

# ii) The main step: Hyper-parameter search, model training and predictions
modelType = 'PROBABILITY'
classPropList = ['instat_class']
pyrPolicy, accuracyMetricString = pyramidingAccuracy(modelType)
print('Model Type: ', modelType, '\n')
print('The pyramiding policy: ', pyrPolicy, '\n')
print('The accuracy metric to be used: ', accuracyMetricString, '\n')
pipeline(finalData, 'bootstrapped', classPropList, k, modelType, projectFolder, composite, classifierList, pipelineChoice)

# iii) Assessment of the level of extrapolation vs. interpolation
# 1) univariate int-ext analysis
# Create a feature collection with only the values from the image bands
pyrPolicy, accuracyMetricString = pyramidingAccuracy(modelType)
univarIntExtImage = univariateIntExt(pyrPolicy, covariateList, 'bootstrapped')

# 2) multi-variate
multivariateIntExt(propOfVariance, 'bootstrapped')

# 3) area of applicability
aoa('bootstrapped', covariateList, weight=True, cv=True)
downloadAoaShape()

# iv) Inherent variation of model due to different seed setting
bestModelName, classifier, fullSetPointLocations, categories = mappingPrep('bootstrapped', modelType, classPropList[0])
inherentModelVariation(1, 10, 0.7, classPropList[0], fullSetPointLocations, covariateList, classifier, pyrPolicy, composite, 'REGRESSION')

# v) Spatial leave-one out cross-validation
# Buffer size in meters; as list or as integer
buffer_size = [10000, 50000, 100000, 250000]
for buffer in buffer_size:
    slooCV(finalData, 'bootstrapped', classPropList[0], bestModelName, covariateList, buffer, 'trial_'+str(buffer), loo_cv_wPointRemoval=False, ensemble=False)

# vi) Further evaluations:
#     - optimal threshold for assigning class labels [non-invaded (0), invaded (1)]
#     - AUC
#     - number of predicted non-invaded vs. invaded plots (also per biome)
#     - plot the probability calibration curve
#     - calibrated the final image
performanceEvaluation(finalData)
calibrateFinalImage()

