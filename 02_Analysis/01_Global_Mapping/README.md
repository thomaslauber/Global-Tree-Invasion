# 01_Global_Mapping
## mainPipeline.py
We used the determined invasion status at the GFBI plot locations and a list of covariate layers to train random forest classifiers and generate a global map of invasion probability. 
This pipeline will produce a global map using a repeated prediction approach and associated uncertainty layers to evaluate the accuracy of the predictions.

For detailed description of the analysis, please refer to: 
[Van den Hoogen, J. et al. (2021): A geospatial mapping pipeline for ecologists. bioRxiv.](https://doi.org/10.1101/2021.07.07.451145)

## configurations 
This folder holds all configurations needed for the pipeline to run, such as, for instance, user configurations, classifiers or directory paths. 

## modules
This folder holds the pipeline.py file that contains all functions needed to run the main pipeline: 
1. Data preparation 
2. Cross-Validation
3. Mapping (model training, testing and prediction)
