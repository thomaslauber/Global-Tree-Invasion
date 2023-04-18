# 02_Drivers_of_Invasion

## featureImportance_Faith.py and featureImportance_lgSprich.py
These two scripts build random forest models to predict i) Invasion status and ii) Invasion severity: Colonization and Spread based on a set of covariables (anthropogenic, soil and climatic) and phylogenetic information (either Faith's PD or Species Richness and Mean Nearest Taxon Distance). The purpose of the scripts are not to investigate the predictions but evaluate the importance of each covariable for the model's predictions. This is investigated using the SHAP framework (https://shap.readthedocs.io/en/latest/, https://github.com/slundberg/shap). 

## additionalScripts
Folder that contains a script including some trials for additional SHAP plots. 