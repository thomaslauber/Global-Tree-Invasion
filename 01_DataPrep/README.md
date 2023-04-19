# 01_dataPrep

## 01_dataPrep.R
This script condenses the GFBI dataframe per plot per year. 

## 02_dataJoin.py
This script is written in python, and requires access to Google Earth Engine via the python API (https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api). 

The main purpose of the script is to evaluate the invasion status of each species occurrence in the GFBI data based on information gathered from the GloNAF and Kew database. Initially, the data from GFBI is filtered using BGCI to only contain trees. Then, the unique coordinates of the GFBI sample locations are spatially joined with the geospatial information in both GloNAF and Kew. Using the information from the spatial join, the invasion status for each species at each location in GFBI is extracted from both GloNAF and Kew. For further steps of the analysis all data points with incongruent invasion status for GloNAF and Kew were excluded. 

## 03_FD_*.R & 03_PD_*.R
These scripts build trait dendrograms and phylogenetic trees to calculate FD and analogous PD metrics for native and whole communities. The scripts with '_noupsample' in their name use no upsampling. 

## 04_GFBI_Downsample.R
This script downsamples plots to a number of plots proportional to the land area covered by each of 14 biomes, while conserving as many tropical plots as possible. This is done to account for unequal representation of plots across biomes. 

## 05_Invasion_Prop_BA.R
This script determines the invasion metrics for each plot. 

## 06_Data_*.R and 06_FD_*.R & 06_PD_*.R
These scripts join the repeated data, the data of all plots, as well as FD and PD data for the analyses. The scripts with '_noupsample' in their name use no upsampling.

## 07_IntactSubset.R
This script subsets the data to plots occurring in intact/protected areas.
