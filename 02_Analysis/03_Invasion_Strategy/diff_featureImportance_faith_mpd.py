# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:17:06 2022

@author: niamh
"""


"""
---------------------------------------------------------------------------------------------------------------------
 Invasion strategy (change in MPD): 
    -------------
    Random forest classifier to predict the change in MPD.
    
    Feature importance is computed using shap values
    -------------
"""
classificationVariable = 'd.mpd'
covariables = ['nat.faith',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Coarse_fragments_000cm',
               'SG_Sand_Content_000cm', 'SG_Soil_pH_H2O_000cm']

output_name = 'dMPD_invasionStrategy'
modelCategory = 'regression'
dMPD_data = input_data.loc[input_data['instat_class']==1,[classificationVariable]+covariables]
dMPD_data.columns =  [classificationVariable] + ['Faith\'s PD',
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
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 42)
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





"""
---------------------------------------------------------------------------------------------------------------------
 Invasion strategy (change in MPD): 
    -------------
    Random forest classifier to predict the change in MPD.
    
    Feature importance is computed using shap values
    -------------
"""
classificationVariable = 'd.mpd'
covariables = ['nat.faith',
               'dist.ports', 'popdensity',
               'CHELSA_BIO_Annual_Mean_Temperature', 'CHELSA_BIO_Annual_Precipitation', 
               'SG_Absolute_depth_to_bedrock', 'SG_Clay_Content_000cm', 
               'SG_Soil_pH_H2O_000cm', 'SG_SOC_Content_000cm']

output_name = 'dMPD_invasionStrategy'
modelCategory = 'regression'
#dMPD_data = input_data.loc[input_data['pinvba']>0,[classificationVariable]+covariables+['biome2']]
dMPD_data = trial.loc[trial['pinvba']>0,[classificationVariable]+covariables+['biome2']]
dMPD_data.columns = [classificationVariable] + ['Faith\'s PD', 
                                                 'Distance to Ports', 'Population Density',
               'Annual Mean Temperature', 'Annual Precipitation', 
               'Absolute depth to bedrock', 'Clay Content',
               'Soil pH(H2O)', 'SOC Content']+['biome2']

# Prep data for analysis: i)  labels and targets, convert to array and split the data into training and testing sets
labels = np.array(dMPD_data[[classificationVariable]])
features_df = dMPD_data.drop([classificationVariable], axis = 1)
feature_list = list(features_df.columns)
feature_list.remove('biome2')
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels, test_size = 0.25, random_state = 42)

# get indices in test data which ones are temperate and which tropical 
temperate_idx = test_features.reset_index(drop=True)[test_features.reset_index(drop=True)['biome2']=='Temperate'].index.values
tropical_idx = test_features.reset_index(drop=True)[test_features.reset_index(drop=True)['biome2']=='Tropical'].index.values

train_labels = np.ravel(train_labels[:, [0]], order='C')
test_labels = np.ravel(test_labels[:, [0]], order='C')

train_features = train_features.drop(['biome2'], axis = 1)
test_features = test_features.drop(['biome2'], axis = 1)

# Part 2: Hyper-parameter selection: Grid search to evaluate what parameters to use for random forest model
regression_rf = RandomForestRegressor()
regression_param_grid = { 
    'random_state': [0],
    'n_estimators': [100],#[50, 100, 150],
    'max_features': ['sqrt'],#['auto', 'sqrt'],
    'max_depth' : [7]#[4,5,6,7,8]
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
shap.summary_plot(shap_values.values, pd.DataFrame(data=test_features, columns=feature_list), plot_type="bar", show=False, color='#3182bd', sort=False)
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
            fcc.set_cmap(my_cmap)
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
Shap per biome
"""
# Temperate
temperate_shap_values = explainer(test_features)
temperate_shap_values.values = temperate_shap_values.values[temperate_idx].mean(axis=0)
temperate_shap_values.base_values = temperate_shap_values.base_values[temperate_idx].mean(axis=0)
temperate_shap_values.data = temperate_shap_values.data[temperate_idx].mean(axis=0)
shap.plots.bar(temperate_shap_values, show=False)
plt.savefig(outputFolder + '/' + output_name + '_ShapBarPlot_Temperate.png', dpi=300, bbox_inches='tight')
plt.show()


# Tropical
tropical_shap_values = explainer(test_features)
tropical_shap_values.values = tropical_shap_values.values[tropical_idx].mean(axis=0)
tropical_shap_values.base_values = tropical_shap_values.base_values[tropical_idx].mean(axis=0)
tropical_shap_values.data = tropical_shap_values.data[tropical_idx].mean(axis=0)
shap.plots.bar(tropical_shap_values, show=False)
plt.savefig(outputFolder + '/' + output_name + '_ShapBarPlot_Tropical.png', dpi=300, bbox_inches='tight')
plt.show()
