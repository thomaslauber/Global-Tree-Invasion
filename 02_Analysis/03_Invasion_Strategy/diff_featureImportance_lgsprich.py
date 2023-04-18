# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:53:16 2021
@author: Niamh Malin Robmann, Crowther Lab ETH Zurich

This script builds a random forest model for: 
    
    i) invasion status (occurrence) for the "native" community at sample location
    ii) invasion proportion (severity) for the "native" community at sample location
    iii) basal area invasion proportion (severity) for the "invasive" community at sample location

also including basal area    
"""   
# load packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
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
input_file = 'GFBI_invasion_foranalyses_5.11.22.csv'
path = 'Y:/Niamh/01_currentProjects/Invasion_project/02_Part2_difference/shap_values'
outputPath = 'results'
dt = datetime.datetime.today().strftime('%Y-%m-%d')
outputFolder = path + '/' + outputPath + '/' + dt + '_lgsprich'

os.mkdir(outputFolder)

input_data = pd.read_csv(input_path + '/' + input_file, sep=',')
input_data['instat_class'] = input_data['in.stat'].apply(lambda x: 0 if x=='non-invaded' else 1)
input_data = input_data.dropna(axis=0)

# suggested color gradient using invasion color #44051
# see https://encycolorpedia.com/440154
invasion_colors_ptw = ['#440154', '#63306f', '#82578a', '#a17fa6', '#c0a8c3', '#dfd3e1', '#ffffff']
invasion_colors_wtp = ['#ffffff', '#dfd3e1', '#c0a8c3', '#a17fa6', '#82578a', '#63306f', '#440154']
my_cmap_wtp = LinearSegmentedColormap.from_list('', invasion_colors_wtp)
my_cmap_ptw = LinearSegmentedColormap.from_list('', invasion_colors_ptw)

"""
---------------------------------------------------------------------------------------------------------------------
 Invasion strategy (change in MNTD): 
    -------------
    Random forest classifier to predict the change in MNTD.
    
    Feature importance is computed using shap values
    -------------
"""
classificationVariable = 'd.mntd'
covariables = ['lgsprich',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Soil_pH_H2O_000cm']

output_name = 'dMNTD_invasionStrategy'
modelCategory = 'regression'
dMNTD_data = input_data.loc[input_data['pinvba']>0,[classificationVariable]+covariables]
dMNTD_data.columns = [classificationVariable] + ['Species Richness',
                                                 'Distance to Ports', 'Population Density',
                                                 'Annual Mean Temperature', 'Annual Precipitation', 
                                                 'Absolute depth to bedrock', 
                                                 'Coarse Fragments', 
                                                 'Sand Content', 'Soil pH H2O']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(dMNTD_data[[classificationVariable]])
features = dMNTD_data.drop([classificationVariable], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
features_df = pd.DataFrame(data=features, columns=feature_list)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10, random_state = 42)

train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2,3],
    'min_samples_leaf': [3, 5, 7],
    'n_estimators':  [50],
    'max_features': ['auto'],
    'max_depth' : [2,8],
    'criterion': ['squared_error']
}

model = regression_rf
param_grid = regression_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
print('best parameters: ', params)
print('best score: ', CV_rfc.best_score_)

# Part 3: Model training and predictions. Train and evalute random forest on the training data. Get feature importance from model
rf_model = RandomForestRegressor(random_state=42, max_features=params.get('max_features'), n_estimators = params.get('n_estimators'), max_depth=params.get('max_depth'))
rf_model.fit(train_features, train_labels)
predictions = rf_model.predict(test_features)

MAE = round(metrics.mean_absolute_error(test_labels, predictions),3)
print('Mean Absolute Error:', MAE)
MSE = round(metrics.mean_squared_error(test_labels, predictions),3)
print('Mean Squared Error:', MSE)
RMSE = round(np.sqrt(metrics.mean_squared_error(test_labels, predictions)), 3)
print('Root Mean Squared Error:', RMSE)
r2 = round(metrics.r2_score(test_labels.ravel(), predictions),3)
print('R-squared scores:', r2)

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/minModel_' + output_name + '_classificationReport.csv', sep=';', index=False)
print(class_report)

"""
Feature Importance using shap values: 
"""
explainer = shap.TreeExplainer(rf_model, )
shap_values = explainer(test_features)

# mean shap 
plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=test_features, columns=feature_list), plot_type="bar", show=False, color='#440154', sort=False)
current_handles, current_labels = plt.gca().get_legend_handles_labels()
plt.legend(current_handles, ['invaded', 'not invaded'])
plt.xlabel('Mean absolute SHAP value')
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_meanShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=test_features, columns=feature_list), show=False, sort=False)
plt.xlabel('SHAP value')
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap_wtp)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(10)
# possibility for removing y labels if all three figures next to each other
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_ShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

"""
Shap value tables
"""
shap_df = pd.Series(pd.DataFrame(shap_values.values, columns=feature_list).abs().mean(0), name = 'non-invaded')
shap_df.to_csv(outputFolder + '/' + output_name + '_shap_values.csv', sep=';', index=False)



"""
---------------------------------------------------------------------------------------------------------------------
 Invasion strategy (change in MPD): 
    -------------
    Random forest classifier to predict the change in MPD.
    
    Feature importance is computed using shap values
    -------------
"""
classificationVariable = 'd.mpd'
covariables = ['lgsprich',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Soil_pH_H2O_000cm']


output_name = 'dMPD_invasionStrategy'
modelCategory = 'regression'
dMPD_data = input_data.loc[input_data['pinvba']>0,[classificationVariable]+covariables]
dMPD_data.columns =  [classificationVariable] + ['Species Richness',
                                                 'Distance to Ports', 'Population Density',
                                                 'Annual Mean Temperature', 'Annual Precipitation', 
                                                 'Absolute depth to bedrock', 
                                                 'Coarse Fragments', 
                                                 'Sand Content', 'Soil pH H2O']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(dMPD_data[[classificationVariable]])
features = dMPD_data.drop([classificationVariable], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
features_df = pd.DataFrame(data=features, columns=feature_list)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2,3,5],
    'min_samples_leaf': [1, 3, 5, 7, 10],
    'n_estimators':  [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [2,5,8],
    'criterion': ['squared_error']
}

model = regression_rf
param_grid = regression_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
print('best parameters: ', params)
print('best score: ', CV_rfc.best_score_)

# Part 3: Model training and predictions. Train and evalute random forest on the training data. Get feature importance from model
rf_model = RandomForestRegressor(random_state=42, max_features=params.get('max_features'), n_estimators = params.get('n_estimators'), max_depth=params.get('max_depth'))
rf_model.fit(train_features, train_labels)
predictions = rf_model.predict(test_features)

MAE = round(metrics.mean_absolute_error(test_labels, predictions),3)
print('Mean Absolute Error:', MAE)
MSE = round(metrics.mean_squared_error(test_labels, predictions),3)
print('Mean Squared Error:', MSE)
RMSE = round(np.sqrt(metrics.mean_squared_error(test_labels, predictions)), 3)
print('Root Mean Squared Error:', RMSE)
r2 = round(metrics.r2_score(test_labels.ravel(), predictions),3)
print('R-squared scores:', r2)

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/minModel_' + output_name + '_classificationReport.csv', sep=';', index=False)
print(class_report)

"""
Feature Importance using shap values: 
"""
explainer = shap.TreeExplainer(rf_model, )
shap_values = explainer(test_features)

# mean shap 
plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=test_features, columns=feature_list), plot_type="bar", show=False, color='#440154', sort=False)
current_handles, current_labels = plt.gca().get_legend_handles_labels()
plt.legend(current_handles, ['invaded', 'not invaded'])
plt.xlabel('Mean absolute SHAP value')
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_meanShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=test_features, columns=feature_list), show=False, sort=False)
plt.xlabel('SHAP value')
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap_wtp)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(10)
# possibility for removing y labels if all three figures next to each other
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_ShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

"""
Shap value tables
"""
shap_df = pd.Series(pd.DataFrame(shap_values.values, columns=feature_list).abs().mean(0), name = 'non-invaded')
shap_df.to_csv(outputFolder + '/' + output_name + '_shap_values.csv', sep=';', index=False)
