# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:06:40 2022

@author: niamh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:53:16 2021
@author: Niamh Malin Robmann, Crowther Lab ETH Zurich

This script builds a random forest model for: 
    
    i) invasion status (occurrence) for the "native" community at sample location
    ii) invasion proportion (severity) for the "native" community at sample location
    iii) basal area invasion proportion (severity) for the "native" community at sample location

For each random forest model, shap values are computed. 



!! ATTENTION !!

https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

shap_interaction = explainer.shap_interaction_values(X)
Also try: shap.summary_plot(shap_interaction, X)

From the main interaction plot, create a dependence plot 
with the interesting interactions

shap.dependence_plot(
    ("experience", "degree"),
    shap_interaction, X,
    display_features=X)



"""   

# load packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import shap

from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import datetime
import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from plotnine import *
import plotnine


# define variables
input_path = 'Y:/Niamh/01_currentProjects/Invasion_project/00_inputData'
input_file = 'GFBI_invasion_foranalyses.csv'
path = 'Y:/Niamh/01_currentProjects/Invasion_project/01_Part1_native/shap_values'
outputPath = 'results'
dt = datetime.datetime.today().strftime('%Y-%m-%d')
outputFolder = path + '/' + outputPath + '/' + dt + '_additionalAnalysis'

os.mkdir(outputFolder)

input_data = pd.read_csv(input_path + '/' + input_file, sep=',')
input_data['instat_class'] = input_data['in.stat'].apply(lambda x: 0 if x=='non-invaded' else 1)
input_data = input_data.dropna()

# per biome get a list of indices
biomes_list = input_data.biome.unique()

def getPlotIds(biome):
    return input_data.loc[input_data.biome==biome,'plot_id'].tolist()
    
plotIds = list(map(getPlotIds, biomes_list))
biome_plotId_dict = dict(zip(biomes_list, plotIds))

# subsample input_data
subsample = input_data.sample(1000).reset_index(drop=True)
my_cmap = plt.get_cmap('Blues')


"""
---------------------------------------------------------------------------------------------------------------------
Stage 1: Invasion status (probability --> status): 
    -------------
    Random forest classifier to predict invasion status.
    Due to imbalanced input data, the minority class (invaded) is oversampled using SMOTE.
    
    Feature importance is computed in 3 ways: 
        i) built in feature importance from random forest classifier (scikit-learn)
        ii) permutation importance (scikit-learn) 
        iii) shap values (shap)
    -------------
"""
classificationVariable = 'instat_class'
# 'perCapita.faith', 'nat.faith'
covariables = ['nat.faith', 'nat.mntd',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Saturated_H2O_Content_015cm', 'SG_Clay_Content_000cm',
               'SG_Coarse_fragments_000cm','SG_SOC_Density_005cm',
               'SG_Silt_Content_000cm', 'SG_Soil_pH_KCl_000cm', 
               'CHELSA_BIO_Precipitation_Seasonality', 'CHELSA_BIO_Precipitation_of_Warmest_Quarter']

clean_covariable_names = ['Faith\'s PD', 'Mntd', 
                                                      'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Saturated H20 Content', 'Clay Content',
               'Coarse Fragments', 'SOC Density',
               'Silt Content', 'Soil pH KCL',
               'Precipitation Seasonality', 'Precipitation of Warmest Quarter']

output_name = 'native_invProb'
modelCategory = 'classification'
native_prob_inv = subsample.loc[:,[classificationVariable]+covariables]
native_prob_inv.columns = [classificationVariable] + ['Faith\'s PD', 'Mntd', 
                                                      'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Saturated H20 Content', 'Clay Content',
               'Coarse Fragments', 'SOC Density',
               'Silt Content', 'Soil pH KCL',
               'Precipitation Seasonality', 'Precipitation of Warmest Quarter']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(native_prob_inv[[classificationVariable]])
features = native_prob_inv.drop([classificationVariable], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
features_df = pd.DataFrame(data=features, columns=feature_list)

# Model building: Random forest
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
classification_rf = RandomForestClassifier()
classification_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2, 3],#, 5, 7],
    'min_samples_leaf': [1, 3],#, 10, 20],
    'n_estimators': [50],# 200, 300],
    'max_features': ['auto'],
    'max_depth' : [8],#, 15],
    'criterion': ['gini'],# 'entropy']
}

model = classification_rf
param_grid = classification_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
CV_rfc.fit(train_features, train_labels.ravel())
params = CV_rfc.best_params_
params_df = pd.DataFrame(CV_rfc.best_params_, index=[0])
params_df.to_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')
print('best parameters: ', params)
print('best score: ', CV_rfc.best_score_)

rf_model = RandomForestClassifier(random_state=42, max_features=params.get('max_features'), n_estimators = params.get('n_estimators'), max_depth=params.get('max_depth'))
rf_model.fit(train_features, train_labels.ravel())
predictions = rf_model.predict_proba(test_features)
predictions_df = pd.DataFrame(predictions, columns=['non_invaded', 'invaded'])

"""
Feature Importance using shap values: 
    i) force plots
"""
labels = native_prob_inv[classificationVariable].to_numpy()
features = native_prob_inv[clean_covariable_names]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(features)

# to get only one value in a binary classification
# we only want to look at invaded = 1 --> shap_values.values[[0]][0][13][1]
oneDim_shap_values = shap_values
oneDim_shaps = np.array(list(map(lambda x: np.delete(x, 0, axis=1), oneDim_shap_values.values)))
oneDim_shaps = oneDim_shaps.reshape(1000, 14)
oneDim_shap_values.values = oneDim_shaps

for cov in clean_covariable_names: 
    shap.dependence_plot(cov, oneDim_shap_values.values, features.values, feature_names=features.columns, show=False)
    plt.savefig(outputFolder + '/' + output_name + '_DependencePlt_' + cov + '.png', dpi=300, bbox_inches='tight')
    
"""
Force plot per biomes
biomes: 'TP_CF', 'TP_BL', 'TR_MBL', 'MD_WD', 'TR_DBL', 'TR_CF', 'Boreal'
"""
def forcePlot_Biome(biome_oi):
    plot_ids_oi = biome_plotId_dict.get(biome_oi)
    biome_samples_ids = subsample[subsample['plot_id'].isin(plot_ids_oi)].index.values
    shap_values_oi = shap_values.values[biome_samples_ids]
    shap.plots.force(explainer.expected_value[1], shap_values_oi.mean(axis=0), 
                     round(features.iloc[biome_samples_ids,:].mean(axis=0),2), 
                     plot_cmap=['#B0C4DE', '#191970'],text_rotation=20,
                     matplotlib=True, feature_names=clean_covariable_names, show=False)
    plt.savefig(outputFolder + '/' + output_name + '_ForcePlot_' + biome_oi + '.png', dpi=300, bbox_inches='tight')

for biome in biomes_list:
    forcePlot_Biome(biome)

"""
Scatter plots for covariates
"""

for cov in clean_covariable_names: 
    print(cov)
    for cov2 in clean_covariable_names: 
        print(cov2)
        

