 # -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:53:16 2021
@author: Niamh Malin Robmann, Crowther Lab ETH Zurich
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
outputFolder = path + '/' + outputPath + '/' + dt + '_invasionStrategy_perBiome'

os.mkdir(outputFolder)

input_data = pd.read_csv(input_path + '/' + input_file)
input_data['instat_class'] = input_data['in.stat'].apply(lambda x: 0 if x=='non-invaded' else 1)
input_data = input_data.dropna()
input_data['id'] = input_data.index

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
"""
classificationVariable = 'd.mntd'
covariables = ['nat.faith',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 
               'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Soil_pH_H2O_000cm', 'biome2']


output_name = 'dMNTD_invasionStrategy'
modelCategory = 'regression'
data = input_data.loc[input_data['instat_class']==1,[classificationVariable]+covariables].reset_index(drop=True)
data.columns = [classificationVariable] + ['Faith\'s PD',
                                                 'Distance to Ports', 'Population Density',
                                                 'Annual Mean Temperature', 
                                                 'Annual Precipitation', 
                                                 'Absolute depth to bedrock', 
                                                 'Coarse Fragments', 
                                                 'Sand Content', 'Soil pH H2O', 'Biome']


# split the data into temperate and tropical data sets for analysis
temperate_data = data.loc[data.Biome=='Temperate',:]
tropical_data = data.loc[data.Biome=='Tropical',:]

## Analysis per biome
# 1) Temperate
# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(temperate_data[[classificationVariable]])
features_df = temperate_data.loc[:, temperate_data.columns!=classificationVariable]
feature_list = list(features_df.columns)
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels, test_size = 0.33, random_state = 42)

train_features = train_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

train_features = train_features.drop(['Biome'], axis=1)
test_features = test_features.drop(['Biome'], axis=1)
train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2,3, 5, 8],
    'min_samples_leaf': [3, 5, 7],
    'n_estimators':  [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [2,8,15],
    'criterion': ['squared_error']
}

model = regression_rf
param_grid = regression_param_grid

"""
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
print('best score: ', CV_rfc.best_score_)
"""

# hardcore the best params to not always have to re-run the cv!
params = {'criterion': 'squared_error', 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 100, 'random_state': 42}
kfoldCV_R2 = 0.3041552849622106
print('best parameters: ', params)
print('k-fold best R-squared score: ', kfoldCV_R2)

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

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 
                     'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2,
                     'k-fold CV R-squared': kfoldCV_R2}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/minModel_' + output_name + '_classificationReport.csv', sep=';', index=False)
print(class_report)

"""
Feature Importance using shap values: 
"""
explainer = shap.TreeExplainer(rf_model, )
shap_values = explainer(test_features)

feature_list.remove('Biome')

# mean shap 

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=np.array(test_features), columns=feature_list), plot_type="bar", show=False, color='#440154', sort=False)
current_handles, current_labels = plt.gca().get_legend_handles_labels()
plt.legend(current_handles, ['invaded', 'not invaded'])
plt.xlabel('Mean absolute SHAP value')
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_meanShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=np.array(test_features), columns=feature_list), show=False, sort=False)
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
shap_df = pd.Series(pd.DataFrame(shap_values.values, columns=feature_list).abs().mean(0), name = 'strategy')
shap_df.to_csv(outputFolder + '/' + output_name + '_shap_values_global_testData.csv', sep=';', index=False)

shap_values_all = explainer(data.drop(['Biome', 'd.mntd'], axis=1))
shap_df_all = pd.Series(pd.DataFrame(shap_values_all.values, columns=feature_list).abs().mean(0), name = 'strategy')
shap_df_all.to_csv(outputFolder + '/' + output_name + '_shap_values_global_allData.csv', sep=';', index=False)

"""
Shap values split per biomes
"""
# Temperate
temperate_shap_values = explainer(test_features) 
temperate_shap_values.values = temperate_shap_values.values.mean(axis=0)
temperate_shap_values.base_values = temperate_shap_values.base_values.mean(axis=0)
temperate_shap_values.data = temperate_shap_values.data.mean(axis=0)

shap.plots.bar(temperate_shap_values, show=False)
plt.savefig(outputFolder + '/' +'ShapBarPlot_temperate.png', dpi=300, bbox_inches='tight')
plt.show()


# 1) Tropical
# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(tropical_data[[classificationVariable]])
features_df = tropical_data.loc[:, tropical_data.columns!=classificationVariable]
feature_list = list(features_df.columns)
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels, test_size = 0.33, random_state = 42)

train_features = train_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

train_features = train_features.drop(['Biome'], axis=1)
test_features = test_features.drop(['Biome'], axis=1)
train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2,3, 5, 8],
    'min_samples_leaf': [3, 5, 7],
    'n_estimators':  [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [2,8,15],
    'criterion': ['squared_error']
}

model = regression_rf
param_grid = regression_param_grid

CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
print('best score: ', CV_rfc.best_score_)

# hardcore the best params to not always have to re-run the cv!
params = {'criterion': 'squared_error', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 7, 'min_samples_split': 2, 'n_estimators': 50, 'random_state': 42}
kfoldCV_R2 = -0.18880424394019002
print('best parameters: ', params)
print('k-fold best R-squared score: ', kfoldCV_R2)

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

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 
                     'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2,
                     'k-fold CV R-squared': kfoldCV_R2}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/minModel_' + output_name + '_classificationReport.csv', sep=';', index=False)
print(class_report)

"""
Feature Importance using shap values: 
"""
explainer = shap.TreeExplainer(rf_model, )
shap_values = explainer(test_features)

feature_list.remove('Biome')

# mean shap 

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=np.array(test_features), columns=feature_list), plot_type="bar", show=False, color='#440154', sort=False)
current_handles, current_labels = plt.gca().get_legend_handles_labels()
plt.legend(current_handles, ['invaded', 'not invaded'])
plt.xlabel('Mean absolute SHAP value')
#plt.yticks([])
plt.savefig(outputFolder + '/' + output_name + '_meanShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_values.values, pd.DataFrame(data=np.array(test_features), columns=feature_list), show=False, sort=False)
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
shap_df = pd.Series(pd.DataFrame(shap_values.values, columns=feature_list).abs().mean(0), name = 'strategy')
shap_df.to_csv(outputFolder + '/' + output_name + '_shap_values_global_testData.csv', sep=';', index=False)

shap_values_all = explainer(data.drop(['Biome', 'd.mntd'], axis=1))
shap_df_all = pd.Series(pd.DataFrame(shap_values_all.values, columns=feature_list).abs().mean(0), name = 'strategy')
shap_df_all.to_csv(outputFolder + '/' + output_name + '_shap_values_global_allData.csv', sep=';', index=False)

"""
Shap values split per biomes
"""
# Temperate
tropical_shap_values = explainer(test_features) 
tropical_shap_values.values = tropical_shap_values.values.mean(axis=0)
tropical_shap_values.base_values = tropical_shap_values.base_values.mean(axis=0)
tropical_shap_values.data = tropical_shap_values.data.mean(axis=0)

shap.plots.bar(tropical_shap_values, show=False)
plt.savefig(outputFolder + '/' +'ShapBarPlot_tropical.png', dpi=300, bbox_inches='tight')
plt.show()
