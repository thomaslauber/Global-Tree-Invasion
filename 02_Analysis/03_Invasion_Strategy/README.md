# 03_Invasion_strategy

All scripts are building a random forest model predicting invasion strategy using the difference in MNTD within the communities when invaders are included in calculating phylogenetic diversity (MNTD). There are a series of scripts, either splitting the data according to biomes or not. 

## diff_featureImportance_biomesSplit.py
This script splits the input data into the two biomes for the SHAP analysis, after having already built the random forest model on the entire set. This is preferred as building the random forest on the split data (based on biomes) results in very low accuracy for tropical data. 

## diff_featureImportance_biomesSplit_separateData.py
This script splits the input data into the two biomes and separately builds the random forest model + SHAP analysis on these data sets separately. 

## diff_featureImportance_faith.py
Script building a random forest + SHAP analysis to model dMNTD with faith in covariates set. 

## diff_featureImportance_mpd.py
Script building a random forest + SHAP analysis to model dMPD with faith in covariates set. 

## diff_featureImportance_lsprich.py
Script building a random forest + SHAP analysis to model dMNTD with lgsprich instead of faith in covariates set. 
