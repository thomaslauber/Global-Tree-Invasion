# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:53:16 2021
@author: Niamh Malin Robmann, Crowther Lab ETH Zurich

This script builds a random forest model for predicting: 
    
    i) invasion status (occurrence) for the "native" community at sample location
    ii) invasion proportion (severity) for the "native" community at sample location
    iii) basal area invasion proportion (severity) for the "native" community at sample location

For each random forest model, shap values are computed to understand the driving covariates of the built model.
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
path = 'Y:/Niamh/01_currentProjects/Invasion_project/01_Part1_native/shap_values'
outputPath = 'results'
dt = datetime.datetime.today().strftime('%Y-%m-%d')
outputFolder = path + '/' + outputPath + '/' + dt
#os.mkdir(outputFolder)

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
INVASION DRIVERS
------------------
Stage 1: Invasion status (probability --> status): 
"""
classificationVariable = 'instat_class'
covariables = ['nat.faith', 'nat.mntd',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Silt_Content_000cm', 'SG_Soil_pH_H2O_000cm']
               
output_name = 'native_invProb'
modelCategory = 'classification'
native_prob_inv = input_data.loc[:,[classificationVariable]+covariables]
native_prob_inv.columns = [classificationVariable] + ['Faith\'s PD', 'Mntd', 
                                                      'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Absolute depth to bedrock', 'Coarse Fragments', 
               'Sand Content', 'Silt Content', 'Soil pH H2O']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(native_prob_inv[[classificationVariable]])
features = native_prob_inv.drop([classificationVariable], axis = 1)
feature_list = list(features.columns)
features = np.array(features)
features_df = pd.DataFrame(data=features, columns=feature_list)

# Model building: Random forest
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
classification_rf = RandomForestClassifier()

"""
classification_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 3, 10, 20],
    'n_estimators': [10,50, 100, 300],
    'max_features': ['auto'],
    'max_depth' : [4, 8, 15],
    'criterion': ['gini', 'entropy']
}

model = classification_rf
param_grid = classification_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy')
CV_rfc.fit(train_features, train_labels.ravel())
params = CV_rfc.best_params_
params_df = pd.DataFrame(CV_rfc.best_params_, index=[0])
params_df.to_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')
print('best parameters from grid search: ', params)
print('k-fold best score: ', CV_rfc.best_score_)
"""

kfold_auc = 0.8982649258781386
params = pd.read_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')

rf_model = RandomForestClassifier(random_state=42, max_features=params.max_features[0], n_estimators = params.n_estimators[0], max_depth=params.max_depth[0])
rf_model.fit(train_features, train_labels.ravel())
predictions = rf_model.predict_proba(test_features)
predictions_df = pd.DataFrame(predictions, columns=['non_invaded', 'invaded'])

fpr, tpr, threshold = roc_curve(test_labels, predictions_df.invaded)
df_fpr_tpr = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':threshold})    
gmean = np.sqrt(tpr * (1-fpr))
index = np.argmax(gmean)
thresholdOpt = round(threshold[index], ndigits=4)
gmeanOpt = round(gmean[index], ndigits = 4)
fprOpt = round(fpr[index], ndigits = 4)
tprOpt = round(tpr[index], ndigits=4)
print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

predictions_df['predicted_class'] = predictions_df['invaded'].apply(lambda x: 1 if x >= thresholdOpt else 0)
print(predictions_df['predicted_class'].value_counts())
print(classification_report(test_labels, predictions_df.loc[:,'predicted_class']))
fpr, tpr, threshold = roc_curve(test_labels, predictions_df.loc[:,'predicted_class'])
auc = metrics.auc(fpr, tpr)
df_fpr_tpr = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':threshold})    
print('auc score: ', auc)

model_validation = pd.DataFrame({'method': 'ROC-AUC', 'aucScore': auc, 'gmeansOpt':gmeanOpt, 'optThreshold':thresholdOpt, 'k-fold accuracy score':kfold_auc}, index=[0])
model_validation.to_csv(outputFolder + '/' + output_name + '_modelValidation.csv')

fig, ax = plt.subplots(1, 1)
plt.plot(fpr, tpr)
ax.scatter(x=fprOpt, y=tprOpt, c='r')
plt.annotate('Optimal threshold \n for class: {}'.format(thresholdOpt), (fprOpt, tprOpt))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.savefig(outputFolder + '/' + output_name + '_ROC_Curve.png', dpi=300, bbox_inches='tight')

test_binary_pred = (rf_model.predict_proba(test_features)[:,1] >= thresholdOpt).astype(bool)

"""
Feature Importance using shap values: 
    i) shap values
"""
shap_values = shap.TreeExplainer(rf_model).shap_values(test_features)
shap_values_DF = pd.DataFrame()

plt.figure()
shap.summary_plot(shap_values, pd.DataFrame(data=test_features, columns=feature_list), plot_type="bar", class_inds=[1], show=False, sort=False, color=my_cmap_ptw)
#current_handles, current_labels = plt.gca().get_legend_handles_labels()
#plt.legend(current_handles, ['invaded', 'not invaded'])
plt.gca().get_legend().remove()
plt.xlabel('Mean absolute SHAP value')
plt.savefig(outputFolder + '/' + output_name + '_meanShapImp.png', dpi=300, bbox_inches='tight')
plt.show()

# individual points summary plot
plt.figure()
shap.summary_plot(shap_values[1], pd.DataFrame(data=test_features, columns=feature_list), show=False, sort=False)
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
shap_cat0 = shap_values[0]
shap_cat0 = pd.Series(pd.DataFrame(shap_cat0, columns=feature_list).abs().mean(0), name = 'non-invaded')
shap_cat1 = shap_values[1]
shap_cat1 = pd.Series(pd.DataFrame(shap_cat1, columns=feature_list).abs().mean(0), name = 'invaded')
shap_df = pd.concat([shap_cat0, shap_cat1], axis=1)
shap_df.to_csv(outputFolder + '/' + output_name + '_shap_values.csv', sep=';', index=False)


"""
---------------------------------------------------------------------------------------------------------------------
Stage 2: Colonization proportion of invaded species
"""
classificationVariable = 'propinvasive'

covariables = ['nat.faith', 'nat.mntd',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Silt_Content_000cm', 'SG_Soil_pH_H2O_000cm']

output_name = 'native_invProp'
modelCategory = 'regression'

invasion_prop_nat = input_data.loc[input_data['pinvba']>0,[classificationVariable]+covariables]
invasion_prop_nat.columns = [classificationVariable] + ['Faith\'s PD', 'Mntd', 
                                                      'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Absolute depth to bedrock', 'Coarse Fragments', 
               'Sand Content', 'Silt Content', 'Soil pH H2O']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(invasion_prop_nat[[classificationVariable]])
features_df = invasion_prop_nat.drop([classificationVariable], axis = 1)
feature_list = list(features_df.columns)
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels, test_size = 0.25, random_state = 42)

train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2,5,7],
    'min_samples_leaf': [1, 3, 10],
    'n_estimators':  [50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [2,5,8],
    'criterion': ['squared_error']
}

model = regression_rf

"""
param_grid = regression_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
params_df = pd.DataFrame(CV_rfc.best_params_, index=[0])
params_df.to_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')
print('best parameters: ', params)
print('best score: ', CV_rfc.best_score_)
"""

kFold_rSquared = 0.727633012160384
params = pd.read_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')

# Part 3: Model training and predictions. Train and evalute random forest on the training data. Get feature importance from model
rf_model = RandomForestRegressor(random_state=42, max_features=params.max_features[0], n_estimators = params.n_estimators[0], max_depth=params.max_depth[0])
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

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2, 'k-fold R-squared score': kFold_rSquared}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/' + output_name + '_classificationReport.csv', sep=';', index=False)
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
Stage 3: Spread (basal area)
"""
classificationVariable = 'pinvba'

# nat.faith
covariables = ['nat.faith', 'nat.mntd',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Silt_Content_000cm', 'SG_Soil_pH_H2O_000cm']


output_name = 'native_invPropBa'
modelCategory = 'regression'
invasion_prop_ba_nat = input_data.loc[input_data['pinvba']>0,[classificationVariable]+covariables]
invasion_prop_ba_nat.columns = [classificationVariable] + ['Faith\'s PD', 'Mntd', 
                                                      'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Absolute depth to bedrock', 'Coarse Fragments', 
               'Sand Content', 'Silt Content', 'Soil pH H2O']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(invasion_prop_ba_nat[[classificationVariable]])
features_df = invasion_prop_ba_nat.drop([classificationVariable], axis = 1)
feature_list = list(features_df.columns)
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels, test_size = 0.25, random_state = 42)

train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()


"""
regression_param_grid = { 
    'random_state': [42],
    'min_samples_split': [2, 5, 7],
    'min_samples_leaf': [1, 5, 10],
    'n_estimators':  [10,50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,8],
    'criterion': ['squared_error']
}
model = regression_rf
param_grid = regression_param_grid
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2')
CV_rfc.fit(train_features, train_labels)
params = CV_rfc.best_params_
params_df = pd.DataFrame(CV_rfc.best_params_, index=[0])
params_df.to_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')
print('best parameters: ', params)
print('best score: ', CV_rfc.best_score_)
"""

kFold_rSquared = 0.25491028057779574
params = pd.read_csv(outputFolder + '/' + output_name + '_CV_bestParams.csv')

# Part 3: Model training and predictions. Train and evalute random forest on the training data. Get feature importance from model
rf_model = RandomForestRegressor(random_state=42, max_features=params.max_features[0], n_estimators = params.n_estimators[0], max_depth=params.max_depth[0])
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

class_report_dict = {'predicted Variables': classificationVariable,'Mean Absolute Error': MAE, 'Mean Squared Error': MSE, 'Root Mean Squared Error': RMSE, 'R-squared score': r2, 'k-fold R-squared score': kFold_rSquared}
class_report = pd.DataFrame(class_report_dict, index=[0])
class_report.to_csv(outputFolder + '/' + output_name + '_classificationReport.csv', sep=';', index=False)
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