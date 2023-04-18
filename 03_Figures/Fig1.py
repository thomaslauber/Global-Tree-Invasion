# Figure 1
# code to start ee in qgis
# - plugins -> python console
# - click on show editor
# - load the script from there

import ee
from ee_plugin import Map
import math

def hexGrid(proj, diameter):
  size = ee.Number(diameter).divide(math.sqrt(3))

  coords = ee.Image.pixelCoordinates(proj)
  vals = {
    'x': coords.select("x"),
    'u': coords.select("x").divide(diameter),
    'v': coords.select("y").divide(size),
    'r': ee.Number(diameter).divide(2),
  }
  i = ee.Image().expression("floor((floor(u - v) + floor(x / r))/3)", vals)
  j = ee.Image().expression("floor((floor(u + v) + floor(v - u))/3)", vals)

  cells = i.long().leftShift(32).add(j.long()).rename("hexgrid")
  return cells

unboundedGeo = ee.Geometry.Polygon([-180, 88, 0, 88, 180, 88, 180, -88, 0, -88, -180, -88], None, False);
grid = hexGrid(ee.Projection('EPSG:4326'), 1)
regionImg = ee.Image(0).byte().paint(unboundedGeo, 1)
mask = grid.addBands(regionImg).reduceConnectedComponents(ee.Reducer.max(), "hexgrid", 256)
grid = grid.updateMask(mask)
background = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type2').first().neq(0).selfMask().subtract(1);
Map.addLayer(background,{'palette':["D3D3D3"]},'background')
forestMask = ee.Image("UMD/hansen/global_forest_change_2021_v1_9").select('treecover2000')
threshold = 10
Map.addLayer(forestMask.gt(threshold).selfMask(),{'palette':["#A8A9AD"]},'ForestExtent')
data = ee.FeatureCollection('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping/finalCorrMatrix')
invasion = data.reduceToImage(['instat_class'],ee.Reducer.mean()).add(1).addBands(grid);
meanInvasion = invasion.reduceConnectedComponents(ee.Reducer.mean(), "hexgrid", 256)
Map.addLayer(meanInvasion.updateMask(meanInvasion.gt(0)),{'min': 1, 'max': 2,'palette':["8ED542","440154"]},'points image')
