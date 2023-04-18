# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:51:39 2022
@author: niamh, t3
Pipeline to gather data together:
    GFBI, Glonaf and Kew
    --> instead of using the last year of a plot; use all years
"""
## Load packages
import pandas as pd
import numpy as np
import subprocess
import os
import os.path
from os import path
import time
from datetime import date
import datetime
import ee
ee.Initialize()


# Initiate variable paths and filenames #
#########################################
outputPath = '../../output'
dataPath = '../../data/'
rawDataPath = '../../data/rawData/'
gfbiPath = rawDataPath + 'GFBI'
glonafPath = rawDataPath + 'GloNAF'
kewPath = rawDataPath + 'Kew'
bcgiPath = rawDataPath + 'BCGI'

GFBI_file = gfbiPath + '/GFBI_fixed_plots_final.csv'
# GFBI_file = gfbiPath + '/GFBI_biome_rarefaction_no_upsample.csv'
##### GFBI_file = gfbiPath + '/GFBI_ecoregion_rarefaction_no_upsample.csv'

bucket = 'crowtherlab_gcsb_t3'
formattedBucket = 'gs://' + bucket
GCSB_subfolder = 'GFBI_Glonaf_Kew_multiyear/'

GEE_path = 'projects/crowtherlab/t3/InvasiveSpecies/GFBI_Glonaf_Kew_multiyear/'
polygonPathGEE_Glonaf = 'projects/crowtherlab/t3/InvasiveSpecies/GFBI_Glonaf_Kew/'
polygonName_Glonaf = 'GloNAF_original_region2'
polygonPathGEE_Kew = 'projects/crowtherlab/t3/InvasiveSpecies/GFBI_Glonaf_Kew/'
polygonName_Kew = 'Kew_level3'
kew_prop = 'LEVEL3_COD'


# Initiate bash functions and arguments #
#########################################
# bash function
bashFunction_EarthEngine = '/Users/Thomas/opt/anaconda3/envs/ee/bin/earthengine'

# Initiated function name
bashFunctionGSUtil = '/Users/Thomas/opt/anaconda3/envs/ee/bin/gsutil'

# Specify the arguments to these functions
arglist_preEEUploadTable = ['upload','table']
arglist_preGSUtilUploadFile = ['-m', 'cp', '-r']
arglist_preGSUtilUploadDirectory = ['-m', 'cp', '-r']
assetIDStringPrefix = '--asset_id='
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_DeleteFolder = ['rm','-r']
arglist_DeleteAsset = ['rm']
arglist_listAssets = ['ls']
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_DeleteFolder = [bashFunction_EarthEngine]+arglist_DeleteFolder
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder
bashCommandList_listAssets = [bashFunction_EarthEngine]+arglist_listAssets


#############
# Main code #
#############
# Read in and prep all data #
#############################

# GFBI data: original data to work with
GFBI_data = pd.read_csv(GFBI_file, chunksize=10000)
GFBI_data = pd.concat(GFBI_data)
GFBI_data = GFBI_data[['plot_id', 'avglat', 'avglon', 'accepted_bin', 'year']].drop_duplicates().reset_index(drop=True)
n_uniqueSpp_initialGFBI = GFBI_data['accepted_bin'].unique().shape[0]
print('Number of distinct species names entries in GFBI: ', n_uniqueSpp_initialGFBI)

# GFBI unique plots (uniqueCoords) to join with Glonaf & Kew polygons
GFBI_uniquePlots_File = 'GFBI_plotCoords.csv'
assetName = GFBI_uniquePlots_File[:-4]
uniqueCoords = GFBI_data[['plot_id', 'avglat', 'avglon']].drop_duplicates().reset_index(drop=True)
uniqueCoords.to_csv(dataPath + '/' + GFBI_uniquePlots_File, header = True, index= False)

# Normal clean-up of GFBI data:
    # add "Genus-only" identifier to the data
# plotYear_overview = GFBI_data[['plot_id', 'year']].groupby('plot_id')['year'].unique().reset_index(drop=False)
# plotYear_overview['maxYear'] = plotYear_overview['year'].apply(lambda x: max(x))
# plotYear_comboToKeep = plotYear_overview[['plot_id', 'maxYear']]
# plotYear_comboToKeep['tuple'] = list(zip(plotYear_comboToKeep.plot_id, plotYear_comboToKeep.maxYear))
# GFBI_data_cleanedYear = GFBI_data[GFBI_data[['plot_id', 'year']].apply(tuple, axis=1).isin(plotYear_comboToKeep.tuple)]
GFBI_data_cleaned['name_check'] = GFBI_data.apply(lambda x: 'spp' if len(x['accepted_bin'].split(' '))==2 else 'Genus only', axis=1)

# Glonaf data set, merge Taxon and Region data, drop duplicates
glonafData = pd.read_csv(glonafPath + '/Taxon_x_List_GloNAF_vanKleunenetal2018Ecology.csv', encoding="UTF-16", sep='\t')
glonafRegions = pd.read_csv(glonafPath + '/Region_GloNAF_vanKleunenetal2018Ecology.csv', encoding='latin')[['region_id','OBJIDsic']]
glonafData = glonafData.merge(glonafRegions, how='left', on='region_id')
glonafData = glonafData.drop_duplicates(subset=['standardized_name', 'region_id', 'status','OBJIDsic'])
glonafData = glonafData[['standardized_name', 'region_id', 'status', 'OBJIDsic']]
glonaf_newNames = {'standardized_name':'Glonaf_stdName', 'region_id':'Glonaf_region_id',
       'status': 'Glonaf_inv_status', 'OBJIDsic':'Glonaf_OBJIDsic'}
glonafData = glonafData.rename(columns=glonaf_newNames)

# GCBI tree list: Glonaf contains all Angiosperms, GFBI only trees
BCGI_treesDF = pd.read_csv(bcgiPath + '/BCGI_tree_list.csv', sep=';')
glonafData = glonafData.merge(BCGI_treesDF, how='left', left_on='Glonaf_stdName', right_on='cleaned_name')
glonafData['cleaned_name'] = glonafData['cleaned_name'].replace(np.nan, 'noTree')
glonafData_Trees = glonafData.loc[glonafData['cleaned_name']!='noTree',:][['Glonaf_stdName', 'Glonaf_region_id', 'Glonaf_inv_status','Glonaf_OBJIDsic']]

# Kew data set
kewSppData = pd.read_csv(kewPath + '/checklist_names_tpl.csv')
kewDistribution = pd.read_csv(kewPath + '/checklist_distribution.txt', sep='|')
kewSppData_clean = kewSppData.dropna(subset=['tpl.name'])[['plant_name_id', 'taxon_name', 'tpl.name', 'tpl.taxonomic.status', 'tpl.note']]
kewData_joined = kewSppData_clean.merge(kewDistribution[['plant_name_id', 'area_code_l3', 'introduced', 'extinct', 'location_doubtful']], )
kewData_joined = kewData_joined[['tpl.name', 'tpl.taxonomic.status', 'tpl.note', 'area_code_l3', 'introduced', 'extinct', 'location_doubtful']]
kewData_joined = kewData_joined.rename(columns={'tpl.name':'Kew_stdName', 'tpl.taxonomic.status': 'Kew_taxonomicStatus',
       'tpl.note':'Kew_tplNote', 'area_code_l3':'Kew_level3_cod', 'introduced':'Kew_introduced', 'extinct':'Kew_extinct', 'location_doubtful':'Kew_location_doubtful'})

kewData_joined['Kew_status'] = kewData_joined.apply(lambda x: 'introduced' if x['Kew_introduced']==1 else ('extinct' if x['Kew_extinct']==1 else 'native'), axis=1)
kewData_joined.insert(7, 'Kew_status', 'native')

########################################################
# Map GFBI coords to Glonaf and Kew regions (polygons) #
########################################################
# Upload GFBI plots (unique coords) to GEE
gsutilBashUploadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [dataPath + '/' + GFBI_uniquePlots_File] + [formattedBucket + '/' + GCSB_subfolder]
subprocess.run(gsutilBashUploadList)

while not all(x in subprocess.run([bashFunctionGSUtil, 'ls', formattedBucket + '/' + GCSB_subfolder], stdout=subprocess.PIPE).stdout.decode('utf-8') for x in [GFBI_uniquePlots_File]):
    print('not yet')
    time.sleep(5)

arglist_postEEUploadTable = ['--x_column', 'avglon', '--y_column', 'avglat', '--csv_qualifier', '']
earthEngineUploadTableCommands = [bashFunction_EarthEngine] + arglist_preEEUploadTable + [assetIDStringPrefix + GEE_path + assetName] + [formattedBucket + '/' + GCSB_subfolder + GFBI_uniquePlots_File] + arglist_postEEUploadTable
subprocess.run(earthEngineUploadTableCommands)

# !! Break and wait until upload has finished
count = 1
while count >= 1:
    taskList = [str(i) for i in ee.batch.Task.list()]
    subsetList = [s for s in taskList if assetName in s]
    subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
    count = len(subsubList)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of uploading jobs:', count)
    time.sleep(30)

# Map GFBI plots to the corresponding region in Glonaf and Kew
GFBI_coords = ee.FeatureCollection(GEE_path + assetName)
Glonaf_polygons = ee.FeatureCollection(polygonPathGEE_Glonaf + polygonName_Glonaf)
kew_polygons = ee.FeatureCollection(polygonPathGEE_Kew + polygonName_Kew)
keyslist = kew_polygons.aggregate_array(kew_prop).distinct()
lengthKeys = keyslist.length()
valueslist = ee.List.sequence(0, ee.Number(lengthKeys).subtract(1))
lookup = ee.Dictionary.fromLists(keyslist, valueslist)
lookupReverse = {v: k for k, v in lookup.getInfo().items()}

# Glonaf
Glonaf_img = Glonaf_polygons.reduceToImage(
    properties=['OBJIDsic'],
    reducer=ee.Reducer.first()).int()

Glonaf_keyTable = Glonaf_img.reduceRegions(
    collection=GFBI_coords,
    reducer=ee.Reducer.first(),
    scale=1)

# Export these points to GCSB for the final download
Glonaf_keyTableExport = ee.batch.Export.table.toCloudStorage(
    collection=Glonaf_keyTable,
    description='GFBI_Glonaf_KeyTable_Export',
    bucket=bucket,
    fileNamePrefix=GCSB_subfolder + 'keyTable_GFBI_Glonaf',
    fileFormat='CSV'
)
Glonaf_keyTableExport.start()

# Export these points to the Assets in GEE
Glonaf_keyTableAsset = ee.batch.Export.table.toAsset(
    collection=Glonaf_keyTable,
    description='GFBI_Glonaf_KeyTable_Asset',
    assetId=GEE_path + 'keyTable_GFBI_Glonaf'
)
Glonaf_keyTableAsset.start()

# Kew
def addNameCode(feature):
  thisClassName = feature.get('LEVEL3_COD')
  return feature.set('LEVEL3_COD_NUM', lookup.get(thisClassName))

kew_polygons = kew_polygons.map(addNameCode);

kew_img = kew_polygons.reduceToImage(
    properties=['LEVEL3_COD_NUM'],
    reducer=ee.Reducer.first()).int()

kew_keyTable = kew_img.reduceRegions(
    collection=GFBI_coords,
    reducer=ee.Reducer.first(),
    scale=1)

# Export these points to GCSB for the final download
kew_keyTableExport = ee.batch.Export.table.toCloudStorage(
    collection=kew_keyTable,
    description='GFBI_Kew_KeyTable_Export',
    bucket=bucket,
    fileNamePrefix=GCSB_subfolder + 'keyTable_GFBI_Kew',
    fileFormat='CSV'
)
kew_keyTableExport.start()

# Export these points to the Assets in GEE
kew_keyTableAsset = ee.batch.Export.table.toAsset(
    collection=kew_keyTable,
    description='GFBI_Kew_KeyTable_Asset',
    assetId=GEE_path + 'keyTable_GFBI_Kew'
)
kew_keyTableAsset.start()

# !! Break and wait
count = 1
while count >= 1:
    taskList = [str(i) for i in ee.batch.Task.list()]
    subsetList = [s for s in taskList if 'KeyTable' in s]
    subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
    count = len(subsubList)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of exporting jobs:',
          count)
    time.sleep(30)

downloadBucket = formattedBucket + '/' + GCSB_subfolder + 'keyTable_GFBI_Kew.csv'
downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', downloadBucket, dataPath]
subprocess.run(downloadBucket)

downloadBucket = formattedBucket + '/' + GCSB_subfolder + 'keyTable_GFBI_Glonaf.csv'
downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', downloadBucket, dataPath]
subprocess.run(downloadBucket)

# !! Break and wait
while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
    print('Download to local folder in progress...! ',
          datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    time.sleep(10)
print('Key table is downloaded.')

######################################################
# Join GFBI with Glonaf and Kew using the key tables #
######################################################

####################
# Master key table #
####################
# Glonaf key table
Glonaf_keyTable = pd.read_csv(dataPath + '/' + 'keyTable_GFBI_Glonaf.csv')[['first', 'plot_id']]
Glonaf_keyTable = Glonaf_keyTable.rename(columns={'first':'Glonaf_OBJIDsic', 'plot_id':'plot_id'})
Glonaf_keyTable['Glonaf_OBJIDsic'] = Glonaf_keyTable['Glonaf_OBJIDsic'].replace(np.nan, 999999)
Glonaf_keyTable['Glonaf_OBJIDsic'] = Glonaf_keyTable['Glonaf_OBJIDsic'].astype(int)
Glonaf_keyTable['Glonaf_GFBIplotID_status'] = Glonaf_keyTable['Glonaf_OBJIDsic'].apply(lambda x: 'no_match' if x==int(999999) else 'match')

# Check 1: # and prop. of plots that were not matched to a Glonaf region
print('Initial number of plots', uniqueCoords.shape[0])
print('Current number of plots in key table', Glonaf_keyTable.shape[0])
Glonaf_prop_noMatch = Glonaf_keyTable.loc[Glonaf_keyTable['Glonaf_GFBIplotID_status']=='no_match'].shape[0]/Glonaf_keyTable.shape[0]
print('The number of GFBI plots that can not be assigned to a Glonaf polygon: ', Glonaf_keyTable.loc[Glonaf_keyTable['Glonaf_GFBIplotID_status']=='no_match'].shape[0])
print('The proportion of GFBI plots that can not be assigned to a Glonaf polygon: ', Glonaf_prop_noMatch)

# Kew key table
Kew_keyTable = pd.read_csv(dataPath + '/' + 'keyTable_GFBI_Kew.csv')[['first', 'plot_id']]
Kew_keyTable = Kew_keyTable.rename(columns={'first':'LEVEL3_COD_NUM', 'plot_id':'plot_id'})
Kew_keyTable['LEVEL3_COD_NUM'] = Kew_keyTable['LEVEL3_COD_NUM'].replace(np.nan, 999999)
Kew_keyTable['LEVEL3_COD_NUM'] = Kew_keyTable['LEVEL3_COD_NUM'].astype(int)
Kew_keyTable['LEVEL3_COD'] = Kew_keyTable['LEVEL3_COD_NUM'].apply(lambda x: 'no_match' if x==int(999999) else lookupReverse.get(x))
Kew_keyTable = Kew_keyTable[['plot_id', 'LEVEL3_COD', 'LEVEL3_COD_NUM']]
Kew_keyTable['Kew_GFBIplotID_status'] = Kew_keyTable['LEVEL3_COD_NUM'].apply(lambda x: 'no_match' if x==int(999999) else 'match')

# Check 1: the # of plot_id in keyTable needs to be the same as
# # of plots in unique coords data (plots that were uploaded initially).
print('Initial number of plots', uniqueCoords.shape[0])
print('Current number of plots in key table', Kew_keyTable.shape[0])
n_allplots = Kew_keyTable.shape[0]
n_noMatches = Kew_keyTable.loc[Kew_keyTable['Kew_GFBIplotID_status']=='no_match'].shape[0]
prop_noMatch = n_noMatches/n_allplots
print('The number of GFBI plots that can not be assigned to a Glonaf polygon: ', n_noMatches)
print('The proportion of GFBI plots that can not be assigned to a Glonaf polygon: ', prop_noMatch)

# Create the master key table
keyTable = Glonaf_keyTable.merge(Kew_keyTable, how='left', on='plot_id')
keyTable = keyTable.rename(columns={'plot_id':'GFBI_plot_id'})[['GFBI_plot_id', 'Glonaf_OBJIDsic', 'Glonaf_GFBIplotID_status',
       'LEVEL3_COD', 'LEVEL3_COD_NUM', 'Kew_GFBIplotID_status']]
print('size must be the same as both size of Glonaf and Kew key tables: ', keyTable.shape[0])
print(keyTable.head())

# Now on this one can create summary stats:
# 1. how many of GFBI plots were not matched to Glonaf / Kew
# 2. which plots is it for Glonaf / Kew, is it the same ones?
anyNoMatch = keyTable.loc[(keyTable['Glonaf_GFBIplotID_status']=='no_match')|(keyTable['Kew_GFBIplotID_status']=='no_match'), ['GFBI_plot_id', 'Glonaf_GFBIplotID_status', 'Kew_GFBIplotID_status']]
print('in total ', anyNoMatch.shape[0], ' GFBI plots could not be mapped to either of the two data set polgyons.')
notFoundInBoth = anyNoMatch.loc[anyNoMatch['Glonaf_GFBIplotID_status']==anyNoMatch['Kew_GFBIplotID_status']]
print('The number of GFBI plots not mapped to either Glonaf nor Kew: ', notFoundInBoth.shape[0])
notFoundGlonafOnly = anyNoMatch.loc[(anyNoMatch['Glonaf_GFBIplotID_status']=='no_match')&(anyNoMatch['Kew_GFBIplotID_status']=='match')]
print('The number of GFBI plots not mapped to Glonaf: ', notFoundGlonafOnly.shape[0])
notFoundKewOnly = anyNoMatch.loc[(anyNoMatch['Glonaf_GFBIplotID_status']=='match')&(anyNoMatch['Kew_GFBIplotID_status']=='no_match')]
print('The number of GFBI plots not mapped to Kew: ', notFoundKewOnly.shape[0])
print(notFoundGlonafOnly.shape[0]+notFoundInBoth.shape[0]+notFoundKewOnly.shape[0])

##########################################################
# Join the entire data together:                         #
#   save a master dataframe with ALL entries (nas etc.)  #
#   filter on the following data frame                   #
##########################################################

# add Glonaf and Kew keys to GFBI data
GFBI_key = GFBI_data_cleaned.merge(keyTable, how='left', left_on='plot_id', right_on='GFBI_plot_id').reset_index(drop=True)
GFBI_key = GFBI_key[['GFBI_plot_id', 'avglat', 'avglon', 'accepted_bin', 'name_check', 'year',
       'Glonaf_OBJIDsic', 'Glonaf_GFBIplotID_status', 'LEVEL3_COD',
       'LEVEL3_COD_NUM', 'Kew_GFBIplotID_status']]
# check if there are missing values: plot_id in GFBI was not matched to any plot_id in keyTable
print('any missing values? If yes, sth is wrong.', GFBI_key[GFBI_key['LEVEL3_COD'].isna()]['GFBI_plot_id'].unique().tolist())
print('any missing values? If yes, sth is wrong.', GFBI_key[GFBI_key['Glonaf_OBJIDsic'].isna()]['GFBI_plot_id'].unique().tolist())

# check the data again
print('key table: ', GFBI_key.head())

# Add Glonaf and Kew information #
##################################
# Glonaf
GFBI_master = GFBI_key.merge(glonafData_Trees, how='left', left_on=['Glonaf_OBJIDsic', 'accepted_bin'],
                             right_on=['Glonaf_OBJIDsic', 'Glonaf_stdName']).reset_index(drop=True)
GFBI_master['Glonaf_inv_status'] = GFBI_master.apply(lambda x: 'unknown' if
                                                     x['Glonaf_GFBIplotID_status']=='no_match' else x['Glonaf_inv_status'], axis=1)
GFBI_master['Glonaf_inv_status'] = GFBI_master['Glonaf_inv_status'].replace(np.nan, 'native', regex=True)
GFBI_master['Glonaf_collapsed_status'] = GFBI_master['Glonaf_inv_status'].apply(lambda x: 'invasive' if x in ['naturalized', 'alien'] else x)
GFBI_master['Glonaf_region_id'] = GFBI_master.apply(lambda x: 999999 if x['Glonaf_collapsed_status']=='native' else x['Glonaf_region_id'], axis=1)
GFBI_master['Glonaf_region_id'] = GFBI_master.apply(lambda x: 999999 if x['Glonaf_collapsed_status']=='unknown' else x['Glonaf_region_id'], axis=1)
GFBI_master = GFBI_master.fillna({'Glonaf_stdName':'no_spp_match_in_Glonaf'})

# Kew
GFBI_master = GFBI_master.merge(kewData_joined, how='left', left_on=['accepted_bin', 'LEVEL3_COD'], right_on=['Kew_stdName', 'Kew_level3_cod'])
GFBI_master = GFBI_master.drop(['Kew_taxonomicStatus', 'Kew_tplNote', 'LEVEL3_COD'], axis=1)
GFBI_master['Kew_status'] = GFBI_master.apply(lambda x: 'unknown' if
                                                     x['Kew_GFBIplotID_status']=='no_match' else x['Kew_status'], axis=1)
GFBI_master['Kew_status'] = GFBI_master['Kew_status'].replace(np.nan, 'unknown', regex=True)
GFBI_master['Kew_level3_cod'] = GFBI_master.apply(lambda x: 'no_match' if
                                                     x['Kew_GFBIplotID_status']=='no_match' else x['Kew_level3_cod'], axis=1)
GFBI_master['Kew_introduced'] = GFBI_master.apply(lambda x: 'unknown' if
                                                     x['Kew_GFBIplotID_status']=='no_match' else x['Kew_introduced'], axis=1)
GFBI_master['Kew_extinct'] = GFBI_master.apply(lambda x: 'unknown' if
                                                     x['Kew_GFBIplotID_status']=='no_match' else x['Kew_extinct'], axis=1)
GFBI_master['Kew_location_doubtful'] = GFBI_master.apply(lambda x: 'unknown' if
                                                     x['Kew_GFBIplotID_status']=='no_match' else x['Kew_location_doubtful'], axis=1)
GFBI_master = GFBI_master.fillna({'Kew_stdName':'no_spp_match_in_Kew'})
GFBI_master = GFBI_master.fillna({'Kew_level3_cod':'no_region_match', 'Kew_introduced':'unknown', 'Kew_extinct':'unknown',
       'Kew_location_doubtful':'unknown', 'Kew_status':'unknown'})

# Save the master file
GFBI_master.to_csv(dataPath + '/' + 'GFBI_Glonaf_Kew_Master.csv')



GFBI_master
plotYear_overview = GFBI_master[['GFBI_plot_id', 'year']].groupby('GFBI_plot_id')['year'].unique().reset_index(drop=False)

argmax = np.where(plotYear_overview['year'].str.len() == plotYear_overview['year'].str.len().max())[0]
plotYear_overview.iloc[argmax]
