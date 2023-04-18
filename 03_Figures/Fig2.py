# Figure 2
# code to start ee in qgis
# - plugins -> python console
# - click on show editor
# - load the script from there

import ee
from ee_plugin import Map

wb = ["08306B","08519C","2171B5","4292C6","6BAED6","9ECAE1","C6DBEF","DEEBF7","F7FBFF"]
ygb = ["FFFFD9", "EDF8B1", "C7E9B4", "7FCDBB", "41B6C4", "1D91C0", "225EA8", "253494", "081D58"]
wbr = ee.List(wb).reverse()

background = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type2').first().neq(0).selfMask().subtract(1);
Map.addLayer(background,{'palette':["D3D3D3"]},'background')

forestMask = ee.Image("UMD/hansen/global_forest_change_2021_v1_9").select('treecover2000')
threshold = 10
maskToApply = forestMask.gt(threshold)
aoa = ee.Image('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/extVsInt/areaOfApplicability').updateMask(maskToApply)
DI_threshold = ee.Number(aoa.get('DI_threshold_wFolds'))

BootstrappedImage = ee.Image('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/model/BootstrappedImage_Calibratedinstat_class')
viridis_flipped = ["AADC32","5DC863","27AD81","21908C","2C728E","3B528B","472D7B","440154"]
Map.addLayer(BootstrappedImage.select('Bootstrapped_Mean').updateMask(maskToApply),{'min':0,'max':0.2,'palette':viridis_flipped},'BootstrappedMean')
Map.addLayer(BootstrappedImage.select('Bootstrapped_Mean').updateMask(aoa.lte(DI_threshold)),{'min':0,'max':0.2,'palette':viridis_flipped},'BootstrappedMean_insideAOA')
Map.addLayer(BootstrappedImage.select('Bootstrapped_CoefOfVar').updateMask(maskToApply),{'min':0,'max':0.5,'palette':ygb},'Bootstrapped_CoefOfVar')
Map.addLayer(BootstrappedImage.select('Bootstrapped_CoefOfVar').updateMask(aoa.lte(DI_threshold)),{'min':0,'max':0.5,'palette':ygb},'Bootstrapped_CoefOfVar_insideAOA')

# img = ee.Image('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/model/Map_inherentModelVariation_instat_class')
# Map.addLayer(img.select('instat_class_predicted_stdDev').divide(img.select('instat_class_predicted_mean')).updateMask(maskToApply),{'min':0,'max':0.5,'palette':ygb},'instat_class_predicted_CV')

img = ee.Image('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/' + 'extVsInt/' +'univariate_IntVsExtMap')
ygbr = ["081D58","253494","225EA8","1D91C0","41B6C4","7FCDBB","C7E9B4","EDF8B1","FFFFD9"]
Map.addLayer(img.updateMask(BootstrappedImage.select(0).gt(-99)).updateMask(maskToApply),{'min':0,'max':1,'palette':ygbr},'univariate_IntVsExtMap')
Map.addLayer(img.updateMask(BootstrappedImage.select(0).gt(-99)).updateMask(aoa.lte(DI_threshold)),{'min':0,'max':1,'palette':ygbr},'univariate_IntVsExtMap_insideAOA')

img = ee.Image('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/' + 'extVsInt/' +'PCA_IntExtMap')
Map.addLayer(img.updateMask(BootstrappedImage.select(0).gt(-99)).updateMask(maskToApply),{'min':0,'max':1,'palette':ygbr},'PCA_IntVsExtMap')
Map.addLayer(img.updateMask(BootstrappedImage.select(0).gt(-99)).updateMask(aoa.lte(DI_threshold)),{'min':0,'max':1,'palette':ygbr},'PCA_IntVsExtMap_insideAOA')



