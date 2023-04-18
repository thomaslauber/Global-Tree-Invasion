# This file defines the classifiers and their hyper-parameter to be run the grid search on. 
# You can adapt the hyper-parameters and add additional classifiers. If you do add additional 
# ones, please don't forget to also add it in the variable classifierList at the end of this script.
# The pipeline has only been tested using random forests as classifiers.
# resource: https://bradleyboehmke.github.io/HOML/random-forest.html
from configurations.user_config import *

# 16751 samples
# number of trees: features×10 trees, 30x10 = 300, try those? [50, 100, 300, 500, 1000]
# roughly 30 featuress
# variables per split: typical default values are mtry=p3 (regression) --> 10, try those? [2,5,10,15,20]
# CLASSIFICATION, REGRESSION, PROBABILITY
"""
Smile Random forest parameters:
numberOfTrees: see reference file:///C:/Users/niamh/AppData/Local/Temp/HowManyTreesinaRandomForest.pdf
variablesPerSplit: 2, 4, 6
minLeafPopulation: 20, 50, 100
bagFraction: constant, set to 0.632
maxNodes: 20, 40, 60
seed: constant, set to 0

in python best parameters: 
{'criterion': 'gini', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 100, 'random_state': 42}
"""

# Define a function that produces the combinations of the parameters for grid searching
def gridSearch(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))

# Define a hyper-parameter dictionary
parameterDict = {"variablesPerSplit":[2, 4, 6],
                 "bagFraction":[0.632, 0.8],
                 "minLeafPopulation":[5, 10, 25, 50, 100],
                 "maxNodes":[25, 50, 75, 100]
                 }

# Get a list with all hyper-parameter combinations
listOfParametersToTry = []
for parameters in gridSearch(parameterDict):
    listOfParametersToTry.append(parameters)

# Define the GEE classifiers
classifierList = []
for i in range(0,len(listOfParametersToTry)):
    dictOfParameters = listOfParametersToTry[i]
    classifier = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP'+str(i+1),'c',ee.Classifier.smileRandomForest(
    numberOfTrees=100,
    variablesPerSplit=dictOfParameters.get('variablesPerSplit'),
    bagFraction=dictOfParameters.get('bagFraction'),
    minLeafPopulation=dictOfParameters.get('minLeafPopulation'),
    maxNodes=dictOfParameters.get('maxNodes'),
    seed=0))
    classifierList.append(classifier)

#
#
# rf_VP1 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP1','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=50,
#     variablesPerSplit=2,
#     bagFraction=0.632,
#     minLeafPopulation=20,
#     maxNodes=20,
#     seed=0
# ))
#
# rf_VP2 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP2','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=50,
#     variablesPerSplit=4,
#     bagFraction=0.632,
#     minLeafPopulation=50,
#     maxNodes=40,
#     seed=0
# ))
#
# rf_VP3 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP3','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=50,
#     variablesPerSplit=6,
#     bagFraction=0.632,
#     minLeafPopulation=100,
#     maxNodes=60,
#     seed=0
# ))
#
# rf_VP4 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP4','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=100,
#     variablesPerSplit=2,
#     bagFraction=0.632,
#     minLeafPopulation=20,
#     maxNodes=20,
#     seed=0
# ))
#
# rf_VP5 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP5','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=100,
#     variablesPerSplit=4,
#     bagFraction=0.632,
#     minLeafPopulation=50,
#     maxNodes=40,
#     seed=0
# ))
#
# rf_VP6 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP6','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=100,
#     variablesPerSplit=6,
#     bagFraction=0.632,
#     minLeafPopulation=100,
#     maxNodes=60,
#     seed=0
# ))
#
# rf_VP7 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP7','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=150,
#     variablesPerSplit=2,
#     bagFraction=0.632,
#     minLeafPopulation=20,
#     maxNodes=20,
#     seed=0
# ))
#
# rf_VP8 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP8','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=150,
#     variablesPerSplit=4,
#     bagFraction=0.632,
#     minLeafPopulation=50,
#     maxNodes=40,
#     seed=0
# ))
#
# rf_VP9 = ee.Feature(ee.Geometry.Point([0,0])).set('cName','rf_VP9','c',ee.Classifier.smileRandomForest(
#     numberOfTrees=150,
#     variablesPerSplit=6,
#     bagFraction=0.632,
#     minLeafPopulation=100,
#     maxNodes=60,
#     seed=0
# ))
#
# # Wrap all of the models into a feature collection for function mapping
# classifierList = [rf_VP1,
#                   rf_VP2,
#                 rf_VP3,
#                 rf_VP4,
#                 rf_VP5,
#                 rf_VP6,
#                 rf_VP7,
#                 rf_VP8,
#                 rf_VP9]