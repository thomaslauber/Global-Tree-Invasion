# Figure 1
# code to start ee in qgis
# - plugins -> python console
# - click on show editor
# - load the script from there

import ee
from ee_plugin import Map

# Load the FeatureCollection
fc = ee.FeatureCollection('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/finalCorrMatrix').map(lambda f: f.set('count',1))

# Reduce the FC into an image to calculate spatial means
scale = 250e3
unboundedGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)
fcImg = fc.reduceToImage(['instat_class','count'], ee.Reducer.mean().combine(ee.Reducer.count()))\
            .reproject(crs='EPSG:4326',scale=scale)\
            .addBands(ee.Image.pixelCoordinates(projection='EPSG:4326'))\
            .rename(['mean_instat_class','count','x','y'])
          # .reduceResolution(ee.Reducer.mean().combine(ee.Reducer.count()), False, 64e3)\

# Sample the reduced image
fc_reduced = fcImg.sample(region=unboundedGeometry, scale=scale, projection='EPSG:4326', numPixels=1e13, tileScale=16, geometries=True)\

# Filter for different sizes of the points
gt1000 = fc_reduced.filterMetadata('count','greater_than',1000)
gt100 = fc_reduced.filterMetadata('count','greater_than',100).filterMetadata('count','not_greater_than',1000)
gt10 = fc_reduced.filterMetadata('count','greater_than',10).filterMetadata('count','not_greater_than',100)
one = fc_reduced.filterMetadata('count','not_greater_than',10)

background = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type2').first().neq(0).selfMask().subtract(1);
Map.addLayer(background,{'palette':["D3D3D3"]},'background')

viridis = ["440154", "472D7B", "3B528B", "2C728E", "21908C", "27AD81", "5DC863", "AADC32"]
unboundedGeo = ee.Geometry.Polygon([-180, 88, 0, 88, 180, 88, 180, -88, 0, -88, -180, -88], None, False);

img_gt1000 = ee.Image().paint(gt1000,'mean_instat_class').focal_max(5).add(1)
img_gt100 = ee.Image().paint(gt100,'mean_instat_class').focal_max(4).add(1)
img_gt10 = ee.Image().paint(gt10,'mean_instat_class').focal_max(3).add(1)
img_one = ee.Image().paint(one,'mean_instat_class').focal_max(2).add(1)

imageToExport = ee.ImageCollection([
    img_gt1000.rename('b').float(),
    img_gt100.rename('b').float(),
    img_gt10.rename('b').float(),
    img_one.rename('b').float()
]).mosaic()
Map.addLayer(imageToExport, {'min': 1, 'max': 1.5,'palette':["D3D3D3","8ED542","440154"]},'points')