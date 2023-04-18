# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# pipeline.py
# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# This python file contains all helper and data preparation specific functions and the main functions
# needed to run the pipeline on given input data in the following format [Lat, Long, variableOfInterest].
#
# This script is made up by three main parts:
# Helper functions:
#   Non-specific helper functions
#   A1: Data preparation specific helper functions
#   A2: Ext vs. Interpolation specific helper functions
#   A3: Area of Applicability specific helper functions
#   B cross validation specific Helper functions
#   C mapping specific helper functions
# A: Data preparation
# B: cross validation
# C: mapping (model training, testing and prediction)
# Main script: pipeline
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------------------- #
# Load global variables from the configuration files
# ------------------------------------------------------------------------------------------------------------- #
from configurations.user_config import *
from configurations.classifiers import *


# ------------------------------------------------------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# Non-specific Helper functions:
# ------------------------------------------------------------------------------------------------------------- #
# Function to convert FeatureCollection to Image
def fcToImg(f):
    # Reduce to image, take mean per pixel
    img = sampledFC.reduceToImage(
        properties=[f],
        reducer=ee.Reducer.mean()
    )
    return img


# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
    result = []
    # Fetch data as a list
    values = fc.toList(100000).getInfo()
    # Fetch column names
    BANDS = fc.first().propertyNames().getInfo()
    # Remove system:index if present
    if 'system:index' in BANDS: BANDS.remove('system:index')

    # Convert to data frame
    for item in values:
        values = item['properties']
        row = [str(values[key]) for key in BANDS]
        row = ",".join(row)
        result.append(row)

    df = pd.DataFrame([item.split(",") for item in result], columns=BANDS)
    df.replace('None', np.nan, inplace=True)

    return df


# duplicateCoordsCol(): This function duplicates the lat and long column to upload the table with
# the coordinates as a geometry in Google Earth Engine and additionally keep them as a property.
# INPUT:
#        data = dataframe
#        latColName = the name of the column containing the latitude coordinate
#        longColName = the name of the column containing the longitude coordinate
# OUTPUT:
#        returns the data frame with two new columns ['geo_Lat', 'geo_Long']. These
#        will be used as the geometry for the upload into GEE.
def duplicateCoordsCol(data, latColName, longColName):
    data['geo_Lat'] = data[latColName]
    data['geo_Long'] = data[longColName]
    return data


# uploadDataToGEE: This function will upload an input data frame as an asset to GEE
# INPUT:
#        localPathToData = path to dataframe
#        fileNameList = list of the filenames (even just one)
#        subfolderNameGCSB = Default value is None. If a value is set, this is the name of the subfolder the
#        data is saved into on the Google Cloud Storage bucket during the upload process. If no name is set,
#        the file is just saved into your main folder.
#        subfolderNameGEE = Default value is None. If a value is set, this is the name of the subfolder the
#        data is saved into in GEE assets directory during the upload process. If no name is set,
#        the file is just saved into your main assets folder on GEE.
#        longString = Default value is none. If value is set this will be taken as the long string for the upload
#        into GEE as the geometry. This has to be set together with latString.
#        latString = Default value is none. If value is set this will be taken as the long string for the upload
# OUTPUT:
#        Upload of the input dataframe as an asset to GEE
def uploadDataToGEE(localPathToData, fileNameList, subfolderNameGCSB, subfolderNameGEE, longString, latString):
    # define the bucket for the upload, using the subfolderNameGCSB
    bucket = formattedBucketOI + '/' + projectFolder + '/' + subfolderNameGCSB + '/'
    print(bucket)

    # for each file in the fileNameList upload it to GCSB
    for f in fileNameList:
        # Format the bash call to upload the file to the Google Cloud Storage bucket
        fullLocalPath = localPathToData + '/' + f
        # gsutilBashUploadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [fullLocalPath] + [bucket]
        gsutilBashUploadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [fullLocalPath] + [bucket]

        subprocess.run(gsutilBashUploadList)

    # Wait for the GSUTIL uploading process to finish before moving on
    while not all(x in subprocess.run([bashFunctionGSUtil, 'ls', bucket], stdout=subprocess.PIPE).stdout.decode(
            'utf-8') for x in fileNameList):
        print('Not everything is uploaded...')
        time.sleep(5)
    # print('Everything is uploaded to GCSB; moving on...')

    if subfolderNameGEE != None:
        newFolderGEE = 'projects/' + usernameFolderString + '/' + projectFolder + '/' + subfolderNameGEE
    else:
        newFolderGEE = 'projects/' + usernameFolderString + '/' + projectFolder
    print(newFolderGEE)

    # define the columns of your data signifying latitude and longitude
    arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString]
    # Loop through the file names and upload each of them to Earth Engine
    for f in fileNameList:
        assetName = f[:-4]
        earthEngineUploadTableCommands = [bashFunction_EarthEngine] + arglist_preEEUploadTable + [
            assetIDStringPrefix + newFolderGEE + '/' + assetName] + [bucket + f] + arglist_postEEUploadTable
        print(earthEngineUploadTableCommands)
        subprocess.run(earthEngineUploadTableCommands)
    # print('All files are being ingested.')

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subList = [s for s in taskList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of uploading jobs:',
              count)
        time.sleep(normalWaitTime)


# This function: downloadCloud downloads a file from Google cloud storage bucket into output directory
# INPUT:
#        fileName = 'exampleName.csv'
#        subfolderGCSB = name of the subfolder in the Google Cloud Storage Bucket
#        targetDirectory = 'user/desktop/targetDirectory/
# OUTPUT:
#        downloaded file in targetDirectory
def downloadCloud(fileNameList, subfolderGCSB, targetDirectory):
    for f in fileNameList:
        formattedBucket = 'gs://' + bucketOfInterest + '/' + '/' + projectFolder + '/' + subfolderGCSB + '/' + f
        gsutilBashDownloadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [formattedBucket] + [
            targetDirectory]
        subprocess.run(gsutilBashDownloadList)

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if fileNameList[0] in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)


# getFilenamesFromDirectory():  saves all the filenames in the chosen directory to a vector.
# INPUT:
#        pathToDirectory = path to directory in which the files are located that you want to get the names from.
# OUTPUT:
#        fileNameList = a vector containing all the file names of the files in the specified folder.
def getFilenamesFromDirectory(pathToDirectory):
    fileNameList = [f for f in listdir(pathToDirectory) if isfile(join(pathToDirectory, f))]
    return fileNameList


# removeNaGroupingVariable(): This function removes all samples/rows from your data
# containing a missing value in the grouping variable
# INPUT:
#        data = matrix where you want to remove NAs from
#        groupingVariable = the grouping variable for the imputation
# OUTPUT:
#         the matrix with NAs removed in the grouping variable.
def removeNaGroupingVariable(data, groupingVariable):
    cleanData = data.dropna(subset=[groupingVariable])
    # print('New number of samples/rows in your data: ', cleanData.shape[0], '\n')
    # print('There are ', cleanData[groupingVariable].isnull().sum(), 'missing values in your grouping variable.')
    return cleanData


# dataTypeChangeToCat(): This function changes a data type from numerical to character string object
# with taking care of the transformation from NA to 'nan' to a still being interpreted as missing values.
# INPUT:
#        data = matrix
#        variableToChangeType = a categorical variable that has a numerical type but should be categorical.
# OUTPUT:
#       the same matrix but the variable is now categorical and not numerical anymore.
def dataTypeChangeNumToCat(data, variableToChangeType):
    before = data[variableToChangeType].isnull().sum()

    data[variableToChangeType] = data[variableToChangeType].astype(str)
    data[variableToChangeType] = data[variableToChangeType].replace('nan', np.nan)

    after = data[variableToChangeType].isnull().sum()
    # print(data[variableToChangeType].isnull().sum())
    if before != after:
        print('Something went wrong with the data transformations. Check the transformation again.')

    # print('New data type: ', data[variableToChangeType].dtypes)
    # print(data.shape)
    return data


# pyramidingAccuracy(): According to the model/data type (classification/categorical versus
# regression/continuous), change variables that are used in the rest of the script.
# INPUT:
#       modelType = the model type that is defined in the user_config.py configuration file.
# OUTPUT:
#       pyramidingPolicy = either 'mode' for Classification models or 'mean' for regression models.
#       accuracyString = either 'OveralAccuracy' for Classification models or 'R2' for regression models.
def pyramidingAccuracy(modelType):
    if modelType == 'CLASSIFICATION':
        pyramidingPolicy = 'mode'
        # print("The pyramiding policy will be 'mode'.")
        accuracyMetricString = 'OverallAccuracy'
        # print("The accuracy type used for crossvalidation will be 'overall accuracy'.")
        return pyramidingPolicy, accuracyMetricString
    if modelType == 'PROBABILITY':
        pyramidingPolicy = 'mode'
        # print("The pyramiding policy will be 'mode'.")
        accuracyMetricString = 'OverallAccuracy'
        # print("The accuracy type used for crossvalidation will be 'overall accuracy'.")
        return pyramidingPolicy, accuracyMetricString

    if modelType == 'REGRESSION':
        # print('No need to compute categorical levels!')
        # print("The pyramiding policy will be 'mean'.")
        pyramidingPolicy = 'mean'
        # print("The accuracy type used for cross validation will be 'coefficient of determination'(i.e., R^2).")
        accuracyMetricString = 'R2'
        return pyramidingPolicy, accuracyMetricString


# ------------------------------------------------------------------------------------------------------------- #
# A.2 Data insights Interpolation vs. extrapolaction specific Helper functions:
# ------------------------------------------------------------------------------------------------------------- #
def univariateIntExt(pyrPolicy, covariateList, mode):
    def calculateUnivariateIntExtGEE(id):
        fcOfInterest = ee.FeatureCollection(id)
        fcForMinMax = fcOfInterest.select(covariateList)
        compositeToClassify = ee.Image(composite).select(covariateList)

        # Make a FC with the band names
        fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(
            lambda bandName: ee.Feature(ee.Geometry.Point([0, 0])).set('BandName', bandName)))

        def calcMinMax(f):
            bandBeingComputed = f.get('BandName')
            maxValueToSet = fcForMinMax.reduceColumns(ee.Reducer.minMax(), [bandBeingComputed])
            return f.set('MinValue', maxValueToSet.get('min')).set('MaxValue', maxValueToSet.get('max'))

        # Map function
        fcWithMinMaxValues = ee.FeatureCollection(fcWithBandNames).map(calcMinMax)

        # Make two images from these values (a min and a max image)
        maxValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MaxValue'))
        maxDict = ee.Dictionary.fromLists(covariateList, maxValuesWNulls)
        minValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MinValue'))
        minDict = ee.Dictionary.fromLists(covariateList, minValuesWNulls)
        minImage = minDict.toImage()
        maxImage = maxDict.toImage()

        totalBandsBinary = compositeToClassify.gte(minImage.select(covariateList)).And(
            compositeToClassify.lte(maxImage.select(covariateList)))
        univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(
            compositeToClassify.bandNames().length()).rename('univariate_pct_int_ext')
        return univariate_int_ext_image

    def calculateUnivariateIntExtLocally(id):
        bootstrappedSample = pd.read_csv(outputPath + '/outputs/bootstrapping/' + id.split('/')[-1] + '.csv')
        bootstrappedSample = bootstrappedSample[covariateList]
        compositeToClassify = ee.Image(composite).select(covariateList)
        # Make two images containing min and a max values
        maxValuesWNulls = list(bootstrappedSample.max())
        maxDict = ee.Dictionary.fromLists(covariateList, maxValuesWNulls)
        minValuesWNulls = list(bootstrappedSample.min())
        minDict = ee.Dictionary.fromLists(covariateList, minValuesWNulls)
        minImage = minDict.toImage()
        maxImage = maxDict.toImage()

        totalBandsBinary = compositeToClassify.gte(minImage.select(covariateList)).And(
            compositeToClassify.lte(maxImage.select(covariateList)))
        univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(
            compositeToClassify.bandNames().length()).rename('univariate_pct_int_ext')
        return univariate_int_ext_image

        # Export the image to the assets
        # totalBandsBinaryImageAsset = ee.batch.Export.image.toAsset(
        #     image=totalBandsBinary,
        #     description='univariateIntExttoAsset_binary',
        #     assetId='projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt/' +'univariate_IntVsExtMap_binary',
        #     crs='EPSG:4326',
        #     crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        #     region=unboundedGeometry,
        #     maxPixels=int(1e13),
        #     pyramidingPolicy={".default": pyrPolicy})
        # totalBandsBinaryImageAsset.start()

    if mode == 'bootstrapped':
        # load the bootstrap samples into ONE image collection and set the values as the
        # training collection
        collectionPath = 'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples/BootstrapSample_'

        # Define a function that sets the sample values as the TrainingColl in each
        # image of the resulting image collection
        def setTrainingColl(seedToUse):
            return ee.Image(0).set('TrainingColl', ee.FeatureCollection(collectionPath + pad(seedToUse, 3)))

        # Map the setTrainingColl over all bootstrapped samples
        ICToMap = ee.ImageCollection(list(map(setTrainingColl, seedsToUseForBootstrapping)))

        # Calculate the univariate extrapolation for each bootstrapped sample
        univariate_int_ext_image = ee.ImageCollection(list(
            map(lambda assetId: calculateUnivariateIntExtLocally(assetId),
                list([fc['id'] for fc in ICToMap.aggregate_array('TrainingColl').getInfo()])))).mean()

    else:
        id_fcOfInterest = 'projects/' + usernameFolderString + '/' + projectFolder + '/finalCorrMatrix'
        univariate_int_ext_image = calculateUnivariateIntExtGEE(id_fcOfInterest)

    # Export the image to the assets
    univariateIntExtImageAsset = ee.batch.Export.image.toAsset(
        image=univariate_int_ext_image,
        description='univariateIntExttoAsset',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt/' + 'univariate_IntVsExtMap',
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=unboundedGeometry,
        maxPixels=int(1e13),
        pyramidingPolicy={".default": pyrPolicy})
    univariateIntExtImageAsset.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if 'univariateIntExttoAsset' in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)
    print('Univariate map has been exported to the assets!')

    return univariate_int_ext_image


def assessExtrapolation(fcOfInterest, propOfVariance, compositeToClassify):
    # Compute the mean and standard deviation of each band, then standardize the point data
    meanVector = fcOfInterest.mean()
    stdVector = fcOfInterest.std()
    standardizedData = (fcOfInterest - meanVector) / stdVector

    # Then standardize the composite from which the points were sampled
    meanList = meanVector.tolist()
    stdList = stdVector.tolist()
    bandNames = list(meanVector.index)
    meanImage = ee.Image(meanList).rename(bandNames)
    stdImage = ee.Image(stdList).rename(bandNames)
    standardizedImage = compositeToClassify.subtract(meanImage).divide(stdImage)

    # Run a PCA on the point samples
    pcaOutput = PCA()
    pcaOutput.fit(standardizedData)

    # Save the cumulative variance represented by each PC
    cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4) * 100)

    # Make a list of PC names for future organizational purposes
    pcNames = ['PC' + str(x) for x in range(1, fcOfInterest.shape[1] + 1)]

    # Get the PC loadings as a data frame
    loadingsDF = pd.DataFrame(pcaOutput.components_, columns=[str(x) + '_Loads' for x in bandNames], index=pcNames)

    # Get the original data transformed into PC space
    transformedData = pd.DataFrame(pcaOutput.fit_transform(standardizedData, standardizedData), columns=pcNames)

    # Make principal components images, multiplying the standardized image by each of the eigenvectors
    # Collect each one of the images in a single image collection

    # First step: make an image collection wherein each image is a PC loadings image
    listOfLoadings = ee.List(loadingsDF.values.tolist())
    eePCNames = ee.List(pcNames)
    zippedList = eePCNames.zip(listOfLoadings)

    def makeLoadingsImage(zippedValue):
        return ee.Image.constant(ee.List(zippedValue).get(1)).rename(bandNames).set('PC', ee.List(zippedValue).get(0))

    loadingsImageCollection = ee.ImageCollection(zippedList.map(makeLoadingsImage))

    # Second step: multiply each of the loadings image by the standardized image and reduce it using a "sum"
    # to finalize the matrix multiplication
    def finalizePCImages(loadingsImage):
        PCName = ee.String(ee.Image(loadingsImage).get('PC'))
        return ee.Image(loadingsImage).multiply(standardizedImage).reduce('sum').rename([PCName]).set('PC', PCName)

    principalComponentsImages = loadingsImageCollection.map(finalizePCImages)

    # Choose how many principal components are of interest in this analysis based on amount of
    # variance explained
    numberOfComponents = sum(i < propOfVariance for i in cumulativeVariance) + 1
    print('Number of Principal Components being used:', numberOfComponents)

    # Compute the combinations of the principal components being used to compute the 2-D convex hulls
    tupleCombinations = list(combinations(list(pcNames[0:numberOfComponents]), 2))
    print('Number of Combinations being used:', len(tupleCombinations))

    # Generate convex hulls for an example of the principal components of interest
    cHullCoordsList = list()
    for c in tupleCombinations:
        firstPC = c[0]
        secondPC = c[1]
        outputCHull = ConvexHull(transformedData[[firstPC, secondPC]])
        listOfCoordinates = transformedData.loc[outputCHull.vertices][[firstPC, secondPC]].values.tolist()
        flattenedList = [val for sublist in listOfCoordinates for val in sublist]
        cHullCoordsList.append(flattenedList)

    # Reformat the image collection to an image with band names that can be selected programmatically
    pcImage = principalComponentsImages.toBands().rename(pcNames)

    # Generate an image collection with each PC selected with it's matching PC
    listOfPCs = ee.List(tupleCombinations)
    listOfCHullCoords = ee.List(cHullCoordsList)
    zippedListPCsAndCHulls = listOfPCs.zip(listOfCHullCoords)

    def makeToClassifyImages(zippedListPCsAndCHulls):
        imageToClassify = pcImage.select(ee.List(zippedListPCsAndCHulls).get(0)).set('CHullCoords', ee.List(
            zippedListPCsAndCHulls).get(1))
        classifiedImage = imageToClassify.rename('u', 'v').classify(
            ee.Classifier.spectralRegion([imageToClassify.get('CHullCoords')]))
        return classifiedImage

    classifedImages = ee.ImageCollection(zippedListPCsAndCHulls.map(makeToClassifyImages))
    finalImageToExport = classifedImages.sum().divide(ee.Image.constant(len(tupleCombinations)))

    return ee.Image(finalImageToExport)

#
# # ------------------------------------------------------------------------------------------------------------- #
# # A.3 Data insights Area of Applicability specific Helper functions:
# # ------------------------------------------------------------------------------------------------------------- #
def aoa(mode, covariateList, weight=None, cv=True):

    # Set the scene
    exportingGeometry = unboundedGeometry
    compositeToClassify = ee.Image(composite)
    scale = compositeToClassify.projection().nominalScale().getInfo()

    # Function to convert FeatureCollection to Image
    def fcToImg(f):
        # Reduce to image, take mean per pixel
        img = fcOI.reduceToImage(
            properties=[f],
            reducer=ee.Reducer.first()
        )
        return img

    # Get the feature importance
    featureImportance = pd.read_csv(outputPath + '/outputs/mapping/featureImportances' + classPropList[0] + '.csv')
    featureImportance = pd.concat([featureImportance, pd.DataFrame({"Covariates": [cvFoldString],"Feature_Importance": [1]}, index=[max(featureImportance.index)+1])])
    # Convert feature importance to image
    featureImportanceImg = ee.Dictionary.fromLists(list(featureImportance['Covariates']),
                                                   list(featureImportance['Feature_Importance'])).toImage().select(sorted(covariateList+[cvFoldString]))

    # Get the data
    if mode == 'bootstrapped':
        dataFileName = 'finalCorrMatrix_resampled'
    else:
        dataFileName = 'finalCorrMatrix'

    dataOI = pd.read_csv(outputPath + '/outputs/dataPrep/' + dataFileName + '.csv')[sorted(covariateList) + [cvFoldString]]
    fcOI = ee.FeatureCollection('projects/crowtherlab/t3/InvasiveSpecies/InvasionMapping_bootstrapped/' + dataFileName)

    ### Standardization of predictor variables
    ## 1. Compute the mean and standard deviation of each band, then standardize the point data
    meanVector = dataOI.mean(); meanVector['foldID'] = 0
    stdVector = dataOI.std(); stdVector['foldID'] = 1
    standardizedData = (dataOI - meanVector) / stdVector

    ## 2. Standardize the predictor space (composite)
    # Compute the mean and standard deviation of each band, convert to image to make calculations below possible
    # Keep the cvFoldString using mean=0 and stdDev=1
    meanVals = ee.Dictionary.fromLists(
        covariateList,
        ee.List(covariateList).map(lambda covariate: fcOI.aggregate_mean(covariate))).set(cvFoldString, 0).toImage().select(sorted(covariateList+[cvFoldString]))
    stdVals = ee.Dictionary.fromLists(
        covariateList,
        ee.List(covariateList).map(lambda covariate: fcOI.aggregate_total_sd(covariate))).set(cvFoldString, 1).toImage().select(sorted(covariateList+[cvFoldString]))
    # Standardize the predictor space
    standarddizedComposite = compositeToClassify.addBands(ee.Image.constant(1).rename(cvFoldString)).select(sorted(covariateList+[cvFoldString])).subtract(meanVals).divide(stdVals)
    # Standardize the FC
    fc_asImg = ee.ImageCollection(list(map(fcToImg, sorted(covariateList+[cvFoldString])))).toBands().rename(sorted(covariateList+[cvFoldString]))
    standardizedFC_asImg = (fc_asImg.subtract(meanVals)).divide(stdVals)

    ### Weighting of variables
    if weight:
        # Weight data
        weightedStandarddizedData = standardizedData * featureImportance.set_index(['Covariates'])['Feature_Importance']
        weightedStandarddizedFC = standardizedFC_asImg.multiply(featureImportanceImg).sampleRegions(
            collection = fcOI,
            properties = ['Index'],
            scale = scale,
            tileScale = 16,
            geometries = False)
        # Weight the composite
        weightedStandarddizedComposite = standarddizedComposite.multiply(featureImportanceImg)
    else:
        weightedStandarddizedData = standardizedData
        weightedStandarddizedFC = standardizedFC_asImg.sampleRegions(
            collection = fcOI,
            properties = ['Index'],
            scale = scale,
            tileScale = 16,
            geometries = False)
        weightedStandarddizedComposite = standarddizedComposite

    ### Multivariate distance calculation
    # Calculate the average distance in the training data
    distancesBetweenTrainingPoints = pdist(weightedStandarddizedData[covariateList], 'euclidean')
    averageDist = distancesBetweenTrainingPoints.mean()

    # Calculate the minimum distance between all training points
    # this is equivalent to a LOOCV scheme
    distMatrix = squareform(distancesBetweenTrainingPoints)
    np.fill_diagonal(distMatrix, np.nan)
    minDist = np.nanmin(distMatrix, axis=1)
    DIs = minDist/averageDist
    upper_quartile, lower_quartile = np.percentile(DIs, 75), np.percentile(DIs, 25)
    iqr = upper_quartile - lower_quartile
    DI_threshold_LOOCV = upper_quartile + 1.5 * iqr

    ### Dissimilarity index and deriving the area of applicability
    # Calculate the distance of every pixel from the predictor space to the training points
    trainedClassifierEucl = ee.Classifier.minimumDistance('euclidean').train(weightedStandarddizedFC,'Index', covariateList).setOutputMode('REGRESSION')
    distancesInPredictorSpace = weightedStandarddizedComposite.classify(trainedClassifierEucl, 'distanceToNearestTrainingPoint')

    if cv is True:
        # Calculate DI for out-of-fold training points
        listOfFoldFCs = []
        for fold in weightedStandarddizedFC.aggregate_array(cvFoldString).distinct().getInfo():
            # Get the training and validation points per fold
            trainingData = weightedStandarddizedFC.filterMetadata(cvFoldString,'not_equals',fold)
            valData = weightedStandarddizedFC.filterMetadata(cvFoldString,'equals',fold)

            # Train the classifier and classify the validation dataset
            trainedClassifier = ee.Classifier.minimumDistance('euclidean').train(trainingData, 'Index', covariateList).setOutputMode('REGRESSION')
            classifiedValidationData = valData.classify(trainedClassifier,'distanceToNearestTrainingPoint')

            # Divide euclidean distance by mean distance calculated above
            listOfFoldFCs.append(classifiedValidationData.map(lambda f: f.set('DI', ee.Number(f.get('distanceToNearestTrainingPoint')).divide(averageDist))))
        DIs = ee.FeatureCollection(listOfFoldFCs).flatten()

        # Calculate upper and lower quartiles, interquartile range and subsequently the DI threshold
        upper_quartile = DIs.aggregate_array('DI').reduce(ee.Reducer.percentile([75]))
        lower_quartile = DIs.aggregate_array('DI').reduce(ee.Reducer.percentile([25]))
        iqr = ee.Number(upper_quartile).subtract(lower_quartile)
        DI_threshold_wFolds = ee.Number(upper_quartile).add(iqr.multiply(1.5))

    # Add the threshold to the AOA image
    distancesInPredictorSpace = distancesInPredictorSpace.divide(averageDist) \
        .set('DI_threshold_LOOCV', DI_threshold_LOOCV)
    if DI_threshold_wFolds != None:
        distancesInPredictorSpace = distancesInPredictorSpace \
            .set('DI_threshold_wFolds', DI_threshold_wFolds)

    ### Export area of applicability image
    aoaImageAsset = ee.batch.Export.image.toAsset(
        image=distancesInPredictorSpace,
        description='areaOfApplicability',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt/' + 'areaOfApplicability',
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=unboundedGeometry,
        maxPixels=int(1e13),
        pyramidingPolicy={".default": 'mean'})
    aoaImageAsset.start()

def downloadAoaShape():
    # Get the aoa
    forestMask = ee.Image("UMD/hansen/global_forest_change_2021_v1_9").select('treecover2000')
    threshold = 10
    maskToApply = forestMask.gt(threshold)
    aoa = ee.Image('projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt/' + 'areaOfApplicability').updateMask(maskToApply)
    DI_threshold = ee.Number(aoa.get('DI_threshold_wFolds'))
    shp = aoa.gt(DI_threshold).selfMask().addBands(aoa).reduceToVectors(reducer=ee.Reducer.first(), geometry= unboundedGeometry, scale= 100*1e3, eightConnected= True, bestEffort= False, maxPixels= 1e13, tileScale= 16)

    # Export the predictions
    shpExport = ee.batch.Export.table.toCloudStorage(
        collection=shp,
        description='aoa_shp',
        bucket=bucketOfInterest,
        fileNamePrefix=projectFolder + '/aoa_shp/aoa',
        fileFormat='SHP'
    )
    shpExport.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if 'aoa' in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)

    # Download the aoa shapefile
    bucket = formattedBucketOI + '/' + projectFolder + '/aoa_shp/'
    directory = outputPath + '/outputs'
    downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
    subprocess.run(downloadBucket)

    # !! Break and wait
    while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
        print('Download to local folder in progress...! ',
              datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(normalWaitTime)

# ------------------------------------------------------------------------------------------------------------- #
# B cross validation specific Helper functions:
# ------------------------------------------------------------------------------------------------------------- #

def CVSubsampling(corrMatrix, strataDict):
    # Step 1: Bootstrapping and upload
    # Create bootstrapped samples from the original data (correlation Matrix)
    # and upload them to the Google Cloud Storage Bucket
    # Initiate an empty list to store all the file names the will be uploaded
    fileNameList = []

    # For now I choose to do 10 subsamples of the initial data
    # to do the CV. Once, the pipeline is ready
    # I want to test how many are necessary to get a rigorous
    # amouont of variation in the results.
    # The cutoff for data will be 10000 observations
    noSubsamples = 10

    # loop through the seeds list (number of bootstrap samples to be created)
    for n in range(noSubsamples):
        # Subsample the correlation matrix stratified by the chosen stratification
        # variable (either 'Resolve_Biome' or 'Resolve_Ecoregion')
        # The each sample will be of size 10000

        # HERE: either choose the percentage or if less available choose all
        # of this biome
        stratSample = corrMatrix.groupby(stratVariable, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(round((strataDict.get(x.name) / 100) * 10000))),
                               replace=True, random_state=n))

        # Upload folder for GCSB
        uploadBucket = formattedBucketOI + '/' + projectFolder + '/Upload/'

        # define the output folder
        holdingFolder = outputPath + '/outputs/CV'

        # Format the title of the CSV and export it to a local directory
        fileNameHeader = 'CVSubsample_'
        titleOfBootstrapCSV = fileNameHeader + str(n).zfill(3)
        fileNameList.append(titleOfBootstrapCSV)
        fullLocalPath = holdingFolder + '/' + titleOfBootstrapCSV + '.csv'
        stratSample.to_csv(holdingFolder + '/' + titleOfBootstrapCSV + '.csv', index=False)

        # Format the bash call to upload the files to the Google Cloud Storage bucket
        gsutilBashUploadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [fullLocalPath] + [uploadBucket]
        subprocess.run(gsutilBashUploadList)
        # print(titleOfBootstrapCSV + ' uploaded to a GCSB!')

    # Wait for the GSUTIL uploading process to finish before moving on
    while not all(
            x in subprocess.run([bashFunctionGSUtil, 'ls', uploadBucket], stdout=subprocess.PIPE).stdout.decode('utf-8')
            for x in fileNameList):
        print('CV subsamples are being uploaded to GCSB...')
        time.sleep(5)
    print('CV subsamples have been uploaded to GCSB.')

    # Step 2: From GCSB upload to Google Earth Engine into a specific asset folder
    # Loop through the file names and upload each of them to Earth Engine
    # define the columns of your data signifying latitude and longitude
    arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString]
    # Loop through the file names and upload each of them to Earth Engine
    for f in fileNameList:
        assetIDForBootstrapColl = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Simple/CVSubsamples'
        gsStorageFileLocation = uploadBucket
        earthEngineUploadTableCommands = [bashFunction_EarthEngine] + arglist_preEEUploadTable + \
                                         [assetIDStringPrefix + assetIDForBootstrapColl + '/' + f] + \
                                         [gsStorageFileLocation + f + '.csv'] + arglist_postEEUploadTable
        subprocess.run(earthEngineUploadTableCommands)
        # print(f + ' EarthEngine Ingestion started!')

    # print('All files are being ingested.')

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if fileNameHeader in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)
    print('Moving on...')

    return fileNameList


# coefficientOfDetermination(): Define the R^2 function for use with continuous valued models
# (i.e., regression based models)
# INPUT:
#       fcOI = feature collection of interest for which R^2 should be computded
#       propertyOfInterest = the property of interest, input data
#       propertyOfInterest_Predicted = the predicted values for the input data
# OUTPUT:
#       R^2
def coefficientOfDetermination(fcOI, propertyOfInterest, propertyOfInterest_Predicted):
    # Compute the mean of the property of interest
    propertyOfInterestMean = ee.Number(ee.Dictionary(
        ee.FeatureCollection(fcOI).select([propertyOfInterest]).
        reduceColumns(ee.Reducer.mean(), [propertyOfInterest])).get('mean'))

    # Compute the total sum of squares
    def totalSoSFunction(f):
        return f.set('Difference_Squared', ee.Number(ee.Feature(f).get(propertyOfInterest)).
                     subtract(propertyOfInterestMean).pow(ee.Number(2)))

    totalSumOfSquares = ee.Number(ee.Dictionary(
        ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).
        reduceColumns(ee.Reducer.sum(), ['Difference_Squared'])).get('sum'))

    # Compute the residual sum of squares
    def residualSoSFunction(f):
        return f.set('Residual_Squared', ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(
            ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))

    residualSumOfSquares = ee.Number(ee.Dictionary(
        ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).
        reduceColumns(ee.Reducer.sum(), ['Residual_Squared'])).get('sum'))

    # Finalize the calculation
    r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))

    return ee.Number(r2)


# ------------------------------------------------------------------------------------------------------------- #
# C mapping specific helper functions:
# ------------------------------------------------------------------------------------------------------------- #

# pad(): This function pads numbers with leading zeroes for formatting purposes.
# INPUT:
#        num = input number, to which zeros should be added
#        size = defines how many zeros should be added
# OUTPUT:
#        num with added 0 in front. [ex.: pad(2, 3) --> 002]
def pad(num, size):
    s = str(num) + ""
    while (len(s) < size):
        s = "0" + s
    return s


# bestModelNameCV(): this function determines the best model from the CV Results
# uses the asset of the CV results
# INPUT:
#        CVResultsAssetID = the name of the k fold CV results asset
#        accucarcyMetricString = name of the accuracy metric used for the modeling (R'2 for regression and
#        overallAccuracy for classification)
# OUTPUT: the namme of the best model determined by CV
def bestModelNameCV(CVResultsAssetID, accuracyMetricString):
    # get the best model from CV_accuracy_FC
    mostAccurateModelFeature = ee.Feature(
        ee.FeatureCollection(CVResultsAssetID).sort('Mean_' + accuracyMetricString, False).sort(
            'Mean_' + accuracyMetricString, False).first())
    bestModelName = mostAccurateModelFeature.get('cName').getInfo()
    # print('Best Model','\n',bestModelName)
    return bestModelName


# stratificationSampleNumberBasedDict(): This function computes a dictionary
# containing the proportion of samples per stratification variable in the data.
# This dictionary is then used to weight the samples in the bootstrapping process.
# Stratification variable should work with all, but tested with 'Resolve_Biome' and 'Resolve_Ecoregion'
# INPUT:
#        stratVariable = the variable to stratify by (usually 'Resolve_Biome' or 'Resolve_Ecoregion')
#        corrMatrix = the correlation matrix containing Pixel coordinates (Lat and Long), the classification
#        property and the covariables
# OUTPUT:
#        strataDict = a dictionary containing the proportion of samples for each group in the stratification
#        variable
def stratificationSampleNumberBasedDict(stratVariable, corrMatrix):
    # Compute the number of samples per stratification variable
    noOfSamplePerStratVariable = corrMatrix.groupby(stratVariable).size()
    # Compute the proportion of samples per stratification variable
    proportion = pd.DataFrame(noOfSamplePerStratVariable / corrMatrix.shape[0] * 100)
    # turn this into a dictionary
    strataDict = pd.Series(proportion[proportion.columns[0]], index=proportion.index).to_dict()
    return strataDict


# stratificationDictionary(): This function computes the stratification dictionary
# based on the user defined inputs 'stratVariable' and 'stratificationChoice'
# Choices: 'area' or 'numberOfSamples'
def stratificationDictionary(stratificationChoice, stratVariable, corrMatrix):
    if stratificationChoice == 'area':
        # strataDict = stratificationAreaBasedDict()
        strataDict = {
            1.0: 14.900835665820974,
            2.0: 2.941697660221864,
            3.0: 0.526059731441294,
            4.0: 9.56387696566245,
            5.0: 2.865354077500338,
            6.0: 11.519674266872787,
            7.0: 16.26999434439293,
            8.0: 8.047078485979089,
            9.0: 0.861212221078014,
            10.0: 3.623974712557433,
            11.0: 6.063922959332467,
            12.0: 2.5132866428302836,
            13.0: 20.037841544639985,
            14.0: 0.26519072167008
        }
    else:
        strataDict = stratificationSampleNumberBasedDict(stratVariable, corrMatrix)
    return strataDict


# ------------------------------------------------------------------------------------------------------------- #
# Main functions:
# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# A.1 Data preparation:  Main function
# ------------------------------------------------------------------------------------------------------------- #

# INPUT:
# data = this is the clean data containin the original LAT and LONG (coordinates) and the classification property
#        meaning the varible that will be modeled. There can be multiple classification properties in the input data.
#        The format of the input data = file.csv, with at leas three columns [Lat, Long, classProperty]
# covariateList = The list of covariates from the composite you want to use for the model creation. The format should
#                 as follows: ['covariate1', 'covariate2', etc.], the names have to be exactly the same as in the
#                 composite.
# groupingVariable = either 'Resolve_Biome' or 'Resolve_Ecoregion'. This variable allows you to choose by which variable
#                    the data is grouped for the missing value imputation. As imputation by mean across the entire
#                    the globe is not the best way.
# imputeStrategy = This variable decides on how the missing values are imputed: either removed, imputed by univariate
#                  mean or median.
# distanceToFill = This variable sets the distance to be gap filled in the gap filling function. It is set to 10000 as
#                  default.
# noOfDecimals = This variable set the number of decimals the pixel coordinates are being rounded in the aggregation
#                function.
# OUTPUT:
# finalCorrMatrix = The output of this function is the final, cleaned and imputed correlation matrix. In the script
# and in the output folder
def dataPrepFun(data, classPropList, covariateList, imputeStrategy, resamplingStrategy, bootstrapping, groupingVariable,
                modeOfAggregation, oversampling=False, distanceToFill=10000, noOfDecimals=7):
    # """
    # PART 1: Pixel coordinates sampling, gap filling and assignment of blockCV ID
    # Duplicate coordinate columns (for upload to GEE)
    dataDuplicated = duplicateCoordsCol(data, latString, longString)

    # define the directory to save the data in
    directory = outputPath + '/outputs/dataPrep'

    # write all of these data tables into the output/dataPrep/data directory
    # dataDuplicated.to_csv(directory + '/dataPixelGapFilling.csv', index=False)
    dataDuplicated.to_csv(directory + '/inputData.csv', index=False, sep=',')
    # """
    # define the list of file names to upload to GEE
    inputData_fileName = ['inputData.csv']
    # """
    # Upload to GEE via GCSB
    uploadDataToGEE(directory, inputData_fileName, 'Upload', 'gapFilling', 'geo_Long', 'geo_Lat')
    # """
    full_composite = ee.Image(composite)

    # Scale of composite
    scale = full_composite.projection().nominalScale().getInfo()

    # Instantiate list of properties to select
    stratVariable = groupingVariable
    fullPropList = covariateList + [stratVariable]

    # Raw dataset as FeatureCollection
    pointCollection = ee.FeatureCollection(
        'projects/' + usernameFolderString + '/' + projectFolder + '/gapFilling/' + inputData_fileName[0][:-4])
    # Sample composite
    sampledFC = full_composite.select(fullPropList).reduceRegions(
        reducer=ee.Reducer.first(),
        collection=pointCollection,
        scale=scale,
        tileScale=16)

    def fcToImg(f):
        # reduce to image, take mean per pixel
        img = sampledFC.reduceToImage(
            properties=[f],
            reducer=ee.Reducer.mean())
        return img

    def getCoordsAsProp(feat):
        Pixel_Long = ee.Feature(feat).geometry().coordinates().get(0)
        Pixel_Lat = ee.Feature(feat).geometry().coordinates().get(1)
        return ee.Feature(feat).set('Pixel_Long', Pixel_Long).set('Pixel_Lat', Pixel_Lat)

    # Perform pixel aggregation by converting FeatureCollection to Image
    fc_asImg = ee.ImageCollection(list(map(fcToImg, fullPropList))).toBands().rename(fullPropList)

    classProp_Img = ee.Image(sampledFC.reduceToImage(
        properties=classPropList,
        reducer=ee.Reducer.mode()).rename(classPropList[0]))

    fc_asImg = fc_asImg.addBands([classProp_Img])

    # Treat differently if many data points are chosen
    if len(data) > 20000:
        exportGeometry = unboundedGeometry
        fc_asImgId = 'projects/' + usernameFolderString + '/' + projectFolder + '/gapFilling/fc_asImg'
        Export_fc_asImg = ee.batch.Export.image.toAsset(
            image=fc_asImg,
            description='fc_asImg',
            assetId=fc_asImgId,
            crs='EPSG:4326',
            crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
            region=exportGeometry,
            maxPixels=int(1e13),
            pyramidingPolicy={".default": 'mean'})
        Export_fc_asImg.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if 'fc_asImg' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)
        print('Converted FC into Image. Moving on...')

        # Load the exported image
        fc_asImg = ee.Image(fc_asImgId)

        # Add a blockCV foldID
        fc_asImg = ee.Image.cat([fc_asImg,
                                 ee.Image.random(0).reproject(crs='EPSG:4326',scale=ee.Number(blockCvSize).multiply(1e3)).rename('blockCV').multiply(10).floor().add(1).divide(10/k).round()
        ])

        # Sample that image to get pixel values
        fc_agg = fc_asImg.sample(
            region=unboundedGeometry,
            scale=scale,
            projection='EPSG:4326',
            geometries=True)

        # Convert biome and blockCV columns to int
        fc_agg = fc_agg.map(lambda f: f.set(stratVariable, ee.Number(f.get(stratVariable)).toInt())
                                        .set('blockCV', ee.Number(f.get('blockCV')).toInt()))

        # Get the coordinates and export the FC to assign CV folds locally
        fcToExport = fc_agg.map(getCoordsAsProp)

        # Export sampled points for aggregation to Bucket
        corrMatrixExport = ee.batch.Export.table.toCloudStorage(
            collection=fcToExport,
            description='export_finalCorrMatrix_prep',
            bucket=bucketOfInterest,
            fileNamePrefix=projectFolder + '/finalCorrMatrix_prep',
            fileFormat='CSV',
            selectors=fullPropList + ['blockCV', classPropList[0], 'Pixel_Lat', 'Pixel_Long'])
        corrMatrixExport.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if 'finalCorrMatrix_prep' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)
        print('Moving on...')

        directory = outputPath + '/outputs/dataPrep'

        # download the covariable filled data from bucket to local folder
        bucket = formattedBucketOI + '/' + projectFolder + '/finalCorrMatrix_prep.csv'
        downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
        subprocess.run(downloadBucket)

        # Load the prepared final correlation matrix
        finalCorrMatrix_prep = pd.read_csv(directory + '/finalCorrMatrix_prep.csv')

        # Add a randomCV foldID locally to the data; resample dataset if needed
        if resamplingStrategy == 'SMOTE':
            fd = []
            for b in np.unique(finalCorrMatrix_prep['Resolve_Biome']):
                biomeData = finalCorrMatrix_prep[finalCorrMatrix_prep['Resolve_Biome'] == b].sample(frac=1)
                sm = SMOTE(random_state=42, k_neighbors=3)
                if len(biomeData[classPropList[0]].drop_duplicates()) > 1:
                    X_res, y_res = sm.fit_resample(biomeData[covariateList], biomeData[classPropList[0]])
                    X_res['Resolve_Biome'] = b
                    biomeData_res = pd.concat([X_res, y_res], axis=1)
                    biomeData = biomeData_res.merge(biomeData.drop(labels=['Resolve_Biome', classPropList[0]], axis=1),
                                                    how='outer', on=covariateList).sample(frac=1)
                    biomeData['.geo'] = biomeData['.geo'].fillna(
                        '{"geodesic":false,"type":"Point","coordinates":[0,0]}')
                biomeData['foldID'] = [i % k + 1 for i in list(range(0, len(biomeData)))]
                fd.append(biomeData)
        if resamplingStrategy == 'None':
            fd = []
            for b in np.unique(finalCorrMatrix_prep['Resolve_Biome']):
                biomeData = finalCorrMatrix_prep[finalCorrMatrix_prep['Resolve_Biome'] == b].sample(frac=1)
                biomeData['foldID'] = [i % k + 1 for i in list(range(0, len(biomeData)))]
                fd.append(biomeData)
        duplicateCoordsCol(pd.concat(fd), 'Pixel_Lat', 'Pixel_Long').to_csv(
            outputPath + '/outputs/dataPrep/finalCorrMatrix.csv', index=False)

        # Resample the data once without replacement if the data should be bootstrapped
        # The outcome will be a stratified sample of the size of one bootstrapped sample
        if bootstrapping == 'bootstrapped':
            # Compute the stratification dictionary
            corrMatrix = pd.concat(fd)
            stratVariable = 'Resolve_Biome'
            strataDict = stratificationDictionary(stratificationChoice, stratVariable, corrMatrix)
            # Create the stratified sample
            stratSample = corrMatrix.groupby(stratVariable, group_keys=False).apply(
                lambda x: x.sample(n=int(round((strataDict.get(x.name) / 100) * bootstrapModelSize)),
                                   replace=True, random_state=0))
            # Reassign the kfolds
            stratSample = stratSample.drop('foldID', axis=1)
            fd = []
            for b in np.unique(stratSample['Resolve_Biome']):
                biomeData = stratSample[stratSample['Resolve_Biome'] == b].sample(frac=1)
                biomeData['foldID'] = [i % k + 1 for i in list(range(0, len(biomeData)))]
                fd.append(biomeData)
            resampledFd = duplicateCoordsCol(pd.concat(fd), 'Pixel_Lat', 'Pixel_Long')
            resampledFd['Index'] = range(1,len(resampledFd)+1)
            resampledFd.to_csv(outputPath + '/outputs/dataPrep/finalCorrMatrix_resampled.csv', index=False)

        # Upload to GEE via GCSB
        uploadDataToGEE(outputPath + '/outputs/dataPrep', ['finalCorrMatrix.csv', 'finalCorrMatrix_resampled.csv'],
                        'Upload', None, 'geo_Long', 'geo_Lat')
        return print(
            'The clean data for the model correlation matrix is in the local folder and in your GEE assets. You are ready to continue to THE PIPELINE TO RULE THEM ALL.')

    else:
        # Add a blockCV foldID
        fc_asImg = ee.Image.cat([fc_asImg,
                                 ee.Image.random(0).reproject(crs='EPSG:4326',scale=ee.Number(blockCvSize).multiply(1e3)).rename('blockCV').multiply(10).floor().add(1).divide(10/k).round()
        ])

        # And sampling that image to get pixel values
        fc_agg = fc_asImg.sample(
            region=unboundedGeometry,
            scale=scale,
            projection='EPSG:4326',
            geometries=True)

        # Convert biome and blockCV columns to int
        fc_agg = fc_agg.map(lambda f: f.set(stratVariable, ee.Number(f.get(stratVariable)).toInt())
                                        .set('blockCV', ee.Number(f.get('blockCV')).toInt()))

        # Retrieve biome classes present in dataset
        biome_list = fc_agg.aggregate_array(stratVariable).distinct()

        # Assign folds to each feature, stratified by biome
        # Function to add folds stratified per biome
        def assignFolds(biome):
            fc_filtered = fc_agg.filter(ee.Filter.eq(stratVariable, biome))
            cvFoldsToAssign = ee.List.sequence(0, fc_filtered.size()).map(lambda i: ee.Number(i).mod(k).add(1))
            fc_sorted = fc_filtered.randomColumn(seed=biome).sort('random')
            fc_wCVfolds = ee.FeatureCollection(cvFoldsToAssign.zip(fc_sorted.toList(fc_filtered.size())).map(
                lambda f: ee.Feature(ee.List(f).get(1)).set(cvFoldString, ee.List(f).get(0))))

            return fc_wCVfolds

        fcToExport = ee.FeatureCollection(biome_list.map(assignFolds)).flatten()
        fcToExport = fcToExport.map(getCoordsAsProp)

        # Export to assets
        fcOI_exportTask = ee.batch.Export.table.toAsset(
            collection=fcToExport,
            description=classPropList[0] + '_finalCorrMatrix',
            assetId='projects/' + usernameFolderString + '/' + projectFolder + '/finalCorrMatrix'
        )
        fcOI_exportTask.start()

        # Export sampled points for aggregation to Bucket
        corrMatrixExport = ee.batch.Export.table.toCloudStorage(
            collection=fcToExport,
            description='export_finalCorrMatrix',
            bucket=bucketOfInterest,
            fileNamePrefix=projectFolder + '/finalCorrMatrix',
            fileFormat='CSV',
            selectors=fullPropList + ['foldID', classPropList[0], 'Pixel_Lat', 'Pixel_Long', '.geo'])

        corrMatrixExport.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if 'finalCorrMatrix' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)
        print('Moving on...')

        directory = outputPath + '/outputs/dataPrep'

        # download the covariable filled data from bucket to local folder
        bucket = formattedBucketOI + '/' + projectFolder + '/finalCorrMatrix.csv'

        downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
        subprocess.run(downloadBucket)
        return print(
            'The clean data for the model correlation matrix is in the local folder and in your GEE assets. You are ready to continue to THE PIPELINE TO RULE THEM ALL.')


# ------------------------------------------------------------------------------------------------------------- #
#  A.2 Data insight: Interpolation vs. Extrapolation  and inherent model uncertainty
#                    Main functions
# ------------------------------------------------------------------------------------------------------------- #

def multivariateIntExt(propOfVariance, mode):
    compositeImage = ee.Image(composite)
    exportGeometry = unboundedGeometry

    # Initialize the band names as the covariates you chose to use for the modelling
    bandNames = covariateList
    cleanedImage = compositeImage.select(bandNames)
    print('Composite Bands', bandNames)

    if mode == 'bootstrapped':
        listOfFinalImages = []
        for path in sorted(glob.glob(outputPath + '/outputs/bootstrapping/*')):
            importedData = pd.read_csv(path)[bandNames]
            listOfFinalImages.append(assessExtrapolation(importedData, propOfVariance, cleanedImage))
        finalImageToExport = ee.ImageCollection(listOfFinalImages).mean()

    else:
        importedData = pd.read_csv(outputPath + '/outputs/dataPrep/finalCorrMatrix.csv')[bandNames]
        finalImageToExport = assessExtrapolation(importedData, propOfVariance, cleanedImage)

    # Export the interpolation extrapolation map to the assets
    exportToAsset = ee.batch.Export.image.toAsset(
        image=finalImageToExport.toFloat(),
        description='CHull_PCA_IntExt_toAsset',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt/PCA_IntExtMap',
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=exportGeometry,
        maxPixels=int(1e13)
    )
    exportToAsset.start()


"""
Prediction accuracy of our model:
    Create per-pixel mean and standard deviation of an iteration of the best random forest model using different random seeds.
    Random forest models are non-deterministic (as name says random to some degree), setting the seed is important for reproducibility.
    To evaluate how results vary depending on the model itself, establish an iteration of the best model using different seeds in each iteration.
    Average the results in the end to get a mean and standard deviation. This gives us some more insights on how the model predictions vary.
"""


def inherentModelVariation(originRandomSeed, noOfSeedsToTest, trainTestSplit, classProperty, fullSetPointLocations,
                           covariateList, bestClassifier, pyrPolicy, composite, mode):
    import random
    updatedComposite = ee.Image(composite)
    exportGeometry = unboundedGeometry

    fcOI = fullSetPointLocations
    fcOIwithRandom = fcOI.randomColumn('random')
    trainingData = fcOIwithRandom.filter(ee.Filter.lt('random', trainTestSplit))
    testData = fcOIwithRandom.filter(ee.Filter.gte('random', trainTestSplit))
    compositeToClassify = ee.Image(updatedComposite).select(covariateList)
    categoricalLevels = [int(float(n)) for n in
                         list(ee.Dictionary(fcOI.aggregate_histogram(classProperty)).keys().getInfo())]

    def makeIC(seedToUse):
        return ee.Image(0).set('seed', seed)

    # ICToMap = ee.ImageCollection(list(map(makeIC, list(range(0, noOfSeedsToTest)))))

    # compute the scale to use for exports
    scaleToUse = ee.Image(updatedComposite).projection().nominalScale().getInfo()

    # Step 1: Train the model noOfSeedsToTest times (k=number of repetitions, number of random seeds to use for building random forest)
    # create a list/vector of seeds to run the model
    IcList = ee.List([])
    IcList_class = ee.List([])
    accuracyFc = ee.FeatureCollection([])
    seeds = []
    random.seed(originRandomSeed)
    for i in range(0, noOfSeedsToTest):
        n = random.randint(1, 30)
        seeds.append(n)

    for seed in seeds:
        print('current seed: ', seed)
        classifierDict = dict(bestClassifier.getInfo().get('classifier'))
        del classifierDict['type']
        classifier_seed = ee.Classifier.smileRandomForest(numberOfTrees=classifierDict.get('numberOfTrees'),
                                                          variablesPerSplit=classifierDict.get('variablesPerSplit'),
                                                          minLeafPopulation=classifierDict.get('minLeafPopulation'),
                                                          bagFraction=classifierDict.get('bagFraction'),
                                                          maxNodes=classifierDict.get('maxNodes'),
                                                          seed=seed).setOutputMode(mode)

        classifier_seed_class = ee.Classifier.smileRandomForest(numberOfTrees=classifierDict.get('numberOfTrees'),
                                                                variablesPerSplit=classifierDict.get(
                                                                    'variablesPerSplit'),
                                                                minLeafPopulation=classifierDict.get(
                                                                    'minLeafPopulation'),
                                                                bagFraction=classifierDict.get('bagFraction'),
                                                                maxNodes=classifierDict.get('maxNodes'),
                                                                seed=seed).setOutputMode('CLASSIFICATION')

        trainedClassifier = classifier_seed.train(trainingData, classProperty, covariateList)
        trainedClassifier_class = classifier_seed_class.train(trainingData, classProperty, covariateList)

        classifiedImage = compositeToClassify.classify(trainedClassifier, classProperty + '_predicted')
        classifiedImage = classifiedImage.set('seed', seed)
        IcList = IcList.add(classifiedImage)

        classifiedImage_class = compositeToClassify.classify(trainedClassifier_class, classProperty + '_predicted')
        classifiedImage_class = classifiedImage_class.set('seed', seed)
        IcList_class = IcList_class.add(classifiedImage_class)

        testData_classified = testData.classify(trainedClassifier_class, classProperty + '_predicted');
        errorMatrix = testData_classified.errorMatrix(classProperty, classProperty + '_predicted',
                                                      categoricalLevels)

        overallAccuracy = ee.Number(errorMatrix.accuracy())
        consumersAccuracy0 = errorMatrix.consumersAccuracy().get([0, 0])
        consumersAccuracy1 = errorMatrix.consumersAccuracy().get([0, -1])
        consumersAccuracy = ee.List([consumersAccuracy0, consumersAccuracy1])
        producersAccuracy0 = errorMatrix.producersAccuracy().get([0, 0])
        producersAccuracy1 = errorMatrix.producersAccuracy().get([1, 0])
        producersAccuracy = ee.List([producersAccuracy0, producersAccuracy1])

        featToAdd = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(0, 0),
                                                     {'seed': seed, 'overall Accuracy': overallAccuracy})])

        accuracyFc = accuracyFc.merge(featToAdd)

    meanStdDevImage = ee.ImageCollection(IcList).reduce(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True))

    # Step 4: Export the final map and data
    # Export final map to Assets and GCSB
    # Export the image to the assets
    fullSingleImageAsset = ee.batch.Export.image.toAsset(
        image=meanStdDevImage.toFloat(),
        description='Mean_inherentModelVariation_' + classProperty,
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/model/Map_inherentModelVariation_' + classProperty,
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=exportGeometry,
        maxPixels=int(1e13),
        pyramidingPolicy={".default": pyrPolicy})
    fullSingleImageAsset.start()


"""
Spatial leave one out cross validation:
    Create per-pixel mean and standard deviation of an iteration of the best random forest model using different random seeds.
    Random forest models are non-deterministic (as name says random to some degree), setting the seed is important for reproducibility.
    To evaluate how results vary depending on the model itself, establish an iteration of the best model using different seeds in each iteration.
    Average the results in the end to get a mean and standard deviation. This gives us some more insights on how the model predictions vary.

Input:
    - loo_cv_wPointRemoval (default value = False): Skip points that fall outside of sampled range after removing points in buffer zone?
                            NB: This might lead to some points never being tested
    - ensemble (default value = false):

"""


def slooCV(finalData, mode, classProperty, bestModelName, covariateList, buffer_size, trial_name,
           loo_cv_wPointRemoval=False, ensemble=False):
    # Helper function 1: assess whether point is within sampled range
    def WithinRange(f):
        testFeature = f
        # Training FeatureCollection: all samples not within geometry of test feature
        trainFC = fcOI.filter(ee.Filter.geometry(f.geometry()).Not())

        # Make a FC with the band names
        fcWithBandNames = ee.FeatureCollection(
            ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName', bandName)))

        # Helper function 1b: assess whether training point is within sampled range; per band
        def getRange(f):
            bandBeingComputed = f.get('BandName')
            minValue = trainFC.aggregate_min(bandBeingComputed)
            maxValue = trainFC.aggregate_max(bandBeingComputed)
            testFeatureWithinRange = ee.Number(testFeature.get(bandBeingComputed)).gte(ee.Number(minValue)).bitwiseAnd(
                ee.Number(testFeature.get(bandBeingComputed)).lte(ee.Number(maxValue)))
            return f.set('within_range', testFeatureWithinRange)

        # Return value of 1 if all bands are within sampled range
        within_range = fcWithBandNames.map(getRange).aggregate_min('within_range')

        return f.set('within_range', within_range)

    # Helper function 2: Spatial Leave One Out cross-validation function:
    def BLOOcv(f):
        # Get iteration ID
        rep = f.get('rep')

        # Test feature
        testFeature = ee.FeatureCollection(f)

        # Training FeatureCollection: all samples not within geometry of test feature
        trainFC = fcOI.filter(ee.Filter.geometry(testFeature).Not())

        if ensemble == False:
            # Classifier to test: same hyperparameter settings as top model from grid search procedure
            classifier = ee.Classifier(ee.Feature(
                ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

        if ensemble == True:
            # Classifiers to test: top 10 models from grid search hyperparameter tuning
            classifierName = top_10Models.get(rep)
            classifier = ee.Classifier(ee.Feature(
                ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get(
                'c'))

        # Train classifier
        trainedClassifer = classifier.train(trainFC, classProperty, covariateList)

        # Apply classifier
        classified = testFeature.classify(classifier=trainedClassifer, outputName='predicted')

        # Get predicted value
        predicted = classified.first().get('predicted')

        # Set predicted value to feature
        return f.set('predicted', predicted).copyProperties(f)

    # Helper function 3: R2 calculation function
    def calc_accuracy(f):
        # Get iteration ID
        rep = f.get('rep')
        # FeatureCollection holding the buffer radius
        buffer_size = f.get('buffer_size')

        # Sample 1000 validation points from the data
        subsetData = fcOI.randomColumn(seed=rep).sort('random').limit(n_points)

        # Add the buffer around the validation data
        fc_wBuffer = subsetData.map(lambda f: f.buffer(buffer_size))

        # Add the iteration ID to the FC
        fc_toValidate = fc_wBuffer.map(lambda f: f.set('rep', rep))

        if loo_cv_wPointRemoval == True:
            # Remove points not within sampled range
            fc_withinSampledRange = fc_toValidate.map(WithinRange).filter(ee.Filter.eq('within_range', 1))

            # Apply blocked leave one out CV function
            predicted = fc_withinSampledRange.map(BLOOcv)

        if loo_cv_wPointRemoval == False:
            # Apply blocked leave one out CV function
            predicted = fc_toValidate.map(BLOOcv)

        # Calculate R2 value
        categoricalLevels = [int(n) for n in list(
            ee.Dictionary(fcOI.aggregate_histogram(classProperty)).keys().getInfo())]
        outputtedPropName = 'predicted'
        errorMatrix = predicted.errorMatrix(classProperty, outputtedPropName,
                                            categoricalLevels)
        overallAccuracy = ee.Number(errorMatrix.accuracy())
        consumersAccuracy0 = errorMatrix.consumersAccuracy().get([0, 0])
        consumersAccuracy1 = errorMatrix.consumersAccuracy().get([0, -1])
        consumersAccuracy = ee.List([consumersAccuracy0, consumersAccuracy1])
        producersAccuracy0 = errorMatrix.producersAccuracy().get([0, 0])
        producersAccuracy1 = errorMatrix.producersAccuracy().get([1, 0])
        producersAccuracy = ee.List([producersAccuracy0, producersAccuracy1])
        kappaStat = ee.Number(errorMatrix.kappa())

        return f.set('Overall Accuracy', overallAccuracy)\
                .set('Consumers Accuracy 0',consumersAccuracy0)\
                .set('Consumers Accuracy 1',consumersAccuracy1)\
                .set('Producers Accuracy 0', producersAccuracy0)\
                .set('Producers Accuracy 1', producersAccuracy1)\
                .set('Kappa Statistic', errorMatrix.kappa())

    if mode == 'bootstrapped':
        fcOI = ee.FeatureCollection(
            'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples/BootstrapSample_001')
    else:
        fcOI = ee.FeatureCollection('projects/' + usernameFolderString + '/' + projectFolder + '/finalCorrMatrix')

    # Set number of random points to test; be a bit careful with large datasets
    if finalData.shape[0] > 1000:
        n_points = 1000  # Don't increase this value!
    else:
        n_points = finalData.shape[0]

    # Set number of repetitions
    n_reps = 5
    nList = list(range(0, n_reps))

    if isinstance(buffer_size, list):
        # create list with species + thresholds
        mapList = []
        for item in nList:
            mapList = mapList + (list(zip(buffer_size, repeat(item))))

        # Make a feature collection from the buffer sizes list
        fc_toMap = ee.FeatureCollection(ee.List(mapList).map(
            lambda n: ee.Feature(ee.Geometry.Point([0, 0])).set('buffer_size', ee.List(n).get(0)).set('rep',
                                                                                                      ee.List(n).get(
                                                                                                          1))))

    else:
        fc_toMap = ee.FeatureCollection(ee.List(nList).map(
            lambda n: ee.Feature(ee.Geometry.Point([0, 0])).set('buffer_size', buffer_size).set('rep', n)))

    # Calculate R2 across range of buffer sizes
    sloo_cv = fc_toMap.map(calc_accuracy)

    # Export FC to assets
    bloo_cv_fc_export = ee.batch.Export.table.toAsset(
        collection=sloo_cv,
        description=classProperty + '_sloo_cv',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/model/' + 'sloo_cv_results_woExtrapolation_' + trial_name
    )
    bloo_cv_fc_export.start()


# ------------------------------------------------------------------------------------------------------------- #
# B cross validation:  Main function
# ------------------------------------------------------------------------------------------------------------- #

# INPUT:
# corrMatrix = This is the clean and ready to go correlation matrix computed by the dataPrep() function.
# k = the number of folds for the CV
# accuracyMetricString = The accuracy metrix and pyramiding policy.
#                        Regression RF: accuracy metric = R^2, pyramiding policy = mean
#                        Classification RF: accuracy metrix = overall accuracy, pyramiding policy = mode
# subfolderNameGCSBUpload = the name of the subfolder in your Google Cloud Storage Bucket to save your files
#                           for the upload
# subfolderNameGCSBDownload = the name of the subfolder in your Google Cloud Storage Bucket to save your files
#                             for the download
# subfolderNameGEE = the name of the subfolder in your GEE asset's directory to save your GEE assets of the uploaded
#                    files in.
# OUTPUT:
# The outputs of this function are in the outputs directory, as a file.csv with the results of the CV
def CV(mode, corrMatrix, classProperty, classifierList, k, accuracyMetricString, subfolderNameGCSBDownload=None,
            subfolderNameGEE=None):
    # PART 1: classifier Preparation
    # for every classifier in classifier list
    # change the name to 'string' + '_classProperty'
    # Then, for every feature set the name to new name
    # ee.Feature.set('cname', newname)
    newClassifierList = []
    for i in range(len(classifierList)):
        oldName = classifierList[i].get('cName').getInfo()
        newName = classProperty + '_' + oldName
        newClassifier = ee.Feature(ee.Geometry.Point([0, 0])).set('cName', newName, 'c',
                                                                  ee.Classifier(classifierList[i].get('c')))
        newClassifierList.append(newClassifier)

    # Make a list of the CV assignments to use
    kList = list(range(1, k + 1))

    # Make a feature collection from the assignment list
    # !! Note: this is used within the scope of the function below, so this should be defined
    # !! explicitly in order for the computeCVAccuracy function to run
    kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0, 0]))
                                                                .set('Fold', n)))

    # There should be different pathways depending on the size of the input data
    # GEE is able (depending on data size and number of covariables) to roughly work with
    # 10000 observations with around 150 covariates. Thus, if the data is larger than 10000 obseervations
    # the large data pathway is run. This pathway subsamples from the original data and runs repeated CV
    # on multiple subsamples for each classifier. The results are then averaged and the average accuracy is used
    # to chose the optimal classifier.
    # For trial: choose 1000, but originally: 10'000
    if corrMatrix.shape[0] <= 15000 or mode == 'bootstrapped':
        categoricalLevels = []
        # PART 2: CV
        # set GEEsubfolder
        if mode == 'bootstrapped':
            subfolderNameGEE = 'CV_Bootstrapped'

        # title of csv
        titleOfCSV = 'finalCorrMatrix_resampled'
        # asset ID
        assetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/' + titleOfCSV
        # Load the collection with the pre-assigned Fold assignments
        fcOI = ee.FeatureCollection(assetID)

        if (modelType == 'CLASSIFICATION') or (modelType == 'PROBABILITY'):
            # Compute the expected values of the classifier (i.e., the levels of the categorical variable, as integers)
            categoricalLevels = [int(float(n)) for n in
                                 list(ee.Dictionary(fcOI.aggregate_histogram(classProperty)).keys().getInfo())]
            print('categorical levels', categoricalLevels)

        # Define a function to take a feature with a classifier of interest
        def computeCVAccuracy(featureWithClassifier):
            # Pull the classifier from the feature
            if (modelType == 'CLASSIFICATION') or (modelType == 'PROBABILITY'):
                cOI = ee.Classifier(featureWithClassifier.get('c')).setOutputMode('CLASSIFICATION')
            else:
                cOI = ee.Classifier(featureWithClassifier.get('c')).setOutputMode('REGRESSION')

            # Create a function to map through the fold assignments and compute the overall accuracy
            # for all validation folds
            def computeAccuracyForFold(foldFeature):
                # Organize the training and validation data
                foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
                trainingData = fcOI.filterMetadata(cvFoldString, 'not_equals', foldNumber)
                validationData = fcOI.filterMetadata(cvFoldString, 'equals', foldNumber)
                # Train the classifier and classify the validation dataset
                trainedClassifier = cOI.train(trainingData, classProperty, covariateList)
                outputtedPropName = classProperty + '_Predicted'
                classifiedValidationData = validationData.classify(trainedClassifier, outputtedPropName)
                # Create a central if/then statement that determines the type of accuracy values that are returned
                if (modelType == 'CLASSIFICATION') or (modelType == 'PROBABILITY'):
                    # Compute the overall accuracy of the classification
                    errorMatrix = classifiedValidationData.errorMatrix(classProperty, outputtedPropName,
                                                                       categoricalLevels)
                    overallAccuracy = ee.Number(errorMatrix.accuracy())

                    consumersAccuracy0 = errorMatrix.consumersAccuracy().get([0, 0])
                    consumersAccuracy1 = errorMatrix.consumersAccuracy().get([0, -1])
                    consumersAccuracy = ee.List([consumersAccuracy0, consumersAccuracy1])
                    producersAccuracy0 = errorMatrix.producersAccuracy().get([0, 0])
                    producersAccuracy1 = errorMatrix.producersAccuracy().get([1, 0])
                    producersAccuracy = ee.List([producersAccuracy0, producersAccuracy1])
                    kappaStat = ee.Number(errorMatrix.kappa())

                    finalFoldFeature = foldFeature.set(accuracyMetricString, overallAccuracy).set('consumers Accuracy',
                                                                                                  consumersAccuracy).set(
                        'producers Accuracy', producersAccuracy).set('entire producers',
                                                                     errorMatrix.producersAccuracy()).set(
                        'entire consumers', errorMatrix.consumersAccuracy()).set('kappa statistic', errorMatrix.kappa())
                    return finalFoldFeature
                else:
                    # Compute the R^2 of the regression
                    r2ToSet = coefficientOfDetermination(classifiedValidationData, classProperty, outputtedPropName)
                    return foldFeature.set(accuracyMetricString, r2ToSet)

            # Compute the feature to return
            if (modelType == 'CLASSIFICATION') or (modelType == 'PROBABILITY'):
                # Compute the accuracy values of the classifier across all folds
                accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)
                meanAccuracy = accuracyFC.aggregate_mean(accuracyMetricString)
                tsdAccuracy = accuracyFC.aggregate_total_sd(accuracyMetricString)

                producersAccuracy_Mean = ee.Array(accuracyFC.aggregate_array('producers Accuracy')).reduce(
                    ee.Reducer.mean(), [0])
                producersAccuracy_stdDev = ee.Array(accuracyFC.aggregate_array('producers Accuracy')).reduce(
                    ee.Reducer.stdDev(), [0])
                consumersAccuracy_Mean = ee.Array(accuracyFC.aggregate_array('consumers Accuracy')).reduce(
                    ee.Reducer.mean(), [0])
                consumersAccuracy_stdDev = ee.Array(accuracyFC.aggregate_array('consumers Accuracy')).reduce(
                    ee.Reducer.stdDev(), [0])
                kappa_Mean = ee.Array(accuracyFC.aggregate_array('kappa statistic')).reduce(ee.Reducer.mean(), [0]).get(
                    [0])
                kappa_stdDev = ee.Array(accuracyFC.aggregate_array('kappa statistic')).reduce(ee.Reducer.stdDev(),
                                                                                              [0]).get([0])

                featureToReturn = featureWithClassifier.select(['cName']).set('Mean_' + accuracyMetricString,
                                                                              meanAccuracy).set(
                    'StDev_' + accuracyMetricString, tsdAccuracy).set('Producers_Mean_Cat0',
                                                                      producersAccuracy_Mean.get([0, 0])).set(
                    'Producers_Mean_Cat1', producersAccuracy_Mean.get([0, 1])).set('Producers_stdDev_Cat0',
                                                                                   producersAccuracy_stdDev.get(
                                                                                       [0, 0])).set(
                    'Producers_stdDev_Cat1', producersAccuracy_stdDev.get([0, 1])).set('Concumers_Mean_Cat0',
                                                                                       consumersAccuracy_Mean.get(
                                                                                           [0, 0])).set(
                    'Concumers_Mean_Cat1', consumersAccuracy_Mean.get([0, 1])).set('Concumers_stdDev_Cat0',
                                                                                   consumersAccuracy_stdDev.get(
                                                                                       [0, 0])).set(
                    'Concumers_stdDev_Cat1', consumersAccuracy_stdDev.get([0, 1])).set('Kappa_Mean', kappa_Mean).set(
                    'Kappa_stdDev', kappa_stdDev)

            else:
                # Compute the accuracy values of the classifier across all folds
                accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)
                meanAccuracy = accuracyFC.aggregate_mean(accuracyMetricString)
                tsdAccuracy = accuracyFC.aggregate_total_sd(accuracyMetricString)
                featureToReturn = featureWithClassifier.select(['cName']).set('Mean_' + accuracyMetricString,
                                                                              meanAccuracy).set(
                    'StDev_' + accuracyMetricString, tsdAccuracy)

            return featureToReturn

        # !! Export the accuracy FC's individually for memory purposes
        for featureWithClassifier in newClassifierList:
            accuracyFC = ee.FeatureCollection(ee.Feature(computeCVAccuracy(featureWithClassifier)))
            classifierName = featureWithClassifier.get('cName').getInfo()
            finalClassifierFCExport = ee.batch.Export.table.toAsset(
                collection=accuracyFC,
                description=classifierName,
                assetId='projects/' + usernameFolderString + '/' + projectFolder + '/' + subfolderNameGEE + '/' + classifierName)
            finalClassifierFCExport.start()
        print('All classification jobs queud.')

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if '_rf_VP' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)
        print('All classifiers have been tested...')

        # PART 3: export the CV results
        # Create/export a feature collection specifically to hold all of the accuracy values
        cvAccuracyFC = []
        for featureWithClassifier in newClassifierList:
            cvAccuracyFC.append(ee.Feature(ee.FeatureCollection(
                'projects/' + usernameFolderString + '/' + projectFolder + '/' + subfolderNameGEE + '/' + str(
                    featureWithClassifier.get('cName').getInfo())).first()))
        cvAccuracyFC = ee.FeatureCollection(cvAccuracyFC).sort('Mean_' + accuracyMetricString, False)
        cvAccuracyFCExport = ee.batch.Export.table.toAsset(
            collection=cvAccuracyFC,
            description='CV_Accuracy_FC',
            assetId='projects/' + usernameFolderString + '/' + projectFolder + '/' + subfolderNameGEE + '/' +
                    classProperty + '_CV_Accuracy_FC')
        cvAccuracyFCExport.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if classProperty in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)
        print('The summary table has been computed...')

        # Print the full set of accuracy values within the feature collection
        # and write it as a csv file into the bucket
        AllResultsFCExport = ee.batch.Export.table.toCloudStorage(
            collection=cvAccuracyFC,
            description='resultsToBucketAsCSV',
            bucket=bucketOfInterest,
            fileNamePrefix=projectFolder + '/' + subfolderNameGEE + '/CVResults' + '_' + classProperty,
            fileFormat='CSV'
        )
        AllResultsFCExport.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if classProperty in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)

        return print('The CV is finalized.')
    else:
        print('Data is too big.')

# This functions downloads the CV predictions
def getCvPredictions(mode, modelType, classProperty, subfolderGEE):
    # Define a function that calculates the predictions per fold
    def predictionsPerFold(foldNumber, cOI):
        # Organize the training and validation data
        foldNumber = ee.Number(foldNumber)
        trainingData = fcOI.filterMetadata(cvFoldString, 'not_equals', foldNumber)
        validationData = fcOI.filterMetadata(cvFoldString, 'equals', foldNumber)
        # Train the classifier and classify the validation dataset
        trainedClassifier = cOI.train(trainingData, classProperty, covariateList)
        outputtedPropName = classProperty + '_Predicted'
        classifiedValidationData = validationData.classify(trainedClassifier, outputtedPropName)
        return classifiedValidationData

    # Define the asset of CV results
    cvAccuracyFCName = classProperty + '_CV_Accuracy_FC'
    if mode == 'simple':
        CVResultsAssetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Simple/' + cvAccuracyFCName
    else:
        CVResultsAssetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Bootstrapped/' + cvAccuracyFCName
    # Determine the best model
    bestModelName = bestModelNameCV(CVResultsAssetID, accuracyMetricString)
    # remove the classification property name
    bestModelName = bestModelName.replace(classProperty + '_', '')

    # Load the best model using the bestModelName
    if (modelType == 'CLASSIFICATION'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('CLASSIFICATION')
    elif (modelType == 'PROBABILITY'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('PROBABILITY')
    elif (modelType == 'REGRESSION'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('REGRESSION')

    # Load the training points
    assetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/' + 'finalCorrMatrix_resampled'
    fcOI = ee.FeatureCollection(assetID)

    # Make a list of the CV assignments to use
    kList = list(range(1, k + 1))

    # Loop over the list and compute the predictions
    cvPredictions = ee.FeatureCollection(ee.List(kList).map(lambda foldNumber: predictionsPerFold(foldNumber, classifier))).flatten()

    # Export the predictions
    AllResultsFCExport = ee.batch.Export.table.toCloudStorage(
        collection=cvPredictions,
        description='cvPredictionsToBucketAsCSV',
        bucket=bucketOfInterest,
        fileNamePrefix=projectFolder + '/' + subfolderGEE + '/CVPredictions' + '_' + classProperty,
        fileFormat='CSV'
    )
    AllResultsFCExport.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if classProperty in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)

# This functions calculates the AUC from CV predictions
def cvAUC(classProperty):
    # Load the predictions
    cvPredictions = pd.read_csv(outputPath + 'outputs/' + subfolderGEE + '/CVPredictions_' + classProperty + '.csv')

    # Compute AUC and other performance criteria etc.
    plt.figure(0).clf()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    classificationReport = pd.DataFrame()
    for fold in sorted(cvPredictions[cvFoldString].drop_duplicates()):
        # Filter for the foldOI
        foldPreds = cvPredictions.loc[cvPredictions[cvFoldString] == fold]

        # ROC curve for finding the optimal threshold
        fpr, tpr, threshold = roc_curve(foldPreds.loc[:, classProperty],
                                        foldPreds.loc[:, classProperty + '_Predicted'])
        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)
        plt.plot(fpr, tpr, c='black', alpha=0.15)
        # find optimal threshold
        gmean = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmean)
        thresholdOpt = threshold[index]

        # Classification dict
        foldPreds['predicted_class'] = foldPreds[classProperty + '_Predicted'].apply(
            lambda x: 1 if x >= thresholdOpt else 0)
        classificationDict = classification_report(foldPreds.loc[:, classProperty],
                                        foldPreds.loc[:, 'predicted_class'], output_dict=True)
        classificationDF = pd.DataFrame([classificationDict['0'], classificationDict['1']],
                                        columns=['precision', 'recall', 'f1-score', 'support'])
        classificationReport = pd.concat([classificationReport, classificationDF])

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.plot(mean_fpr,mean_tpr,color="b",label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),lw=2,alpha=0.8)
    plt.fill_between(mean_fpr,tprs_lower,tprs_upper,color="b",alpha=0.2,label=r"$\pm$ 1 std. dev.")
    plt.title('Mean ROC curve of cross validation')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.savefig(outputPath + '/outputs/ROC_Curve_CV.pdf')
    plt.show()

    classificationReport = classificationReport.groupby(classificationReport.index).mean()
    classificationReport.to_csv(outputPath + '/outputs/classificationReport_CV.csv')

    return (mean_auc, std_auc)


# ------------------------------------------------------------------------------------------------------------- #
# C Mapping:  Main functions
# ------------------------------------------------------------------------------------------------------------- #

# mappinfPrep(): This function prepares necessary variables and inputs for the mapping process
# INPUT: none
# OUTPUT:
#        bestModelName = name of the best model from the CV results
#        classifier = the specific classifier (best model) from the CV results
#        fullSetPointLocations = the full set of points to predict on
def mappingPrep(mode, modelType, classProperty):
    # Name of the asset
    cvAccuracyFCName = classProperty + '_CV_Accuracy_FC'

    # Use the uploaded clean correlation matrix (with the fold assigned column)
    if mode == 'bootstrapped':
        assetIDCorrMatrix = 'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples/BootstrapSample_001'
        fullSetPointLocations = ee.FeatureCollection(assetIDCorrMatrix)
    else:
        assetIDCorrMatrix = 'projects/' + usernameFolderString + '/' + projectFolder + '/finalCorrMatrix'
        fullSetPointLocations = ee.FeatureCollection(assetIDCorrMatrix)

    # According to the model/data type (classification/categorical versus regression/continuous), change variables that are used in the rest of the script
    if modelType == 'CLASSIFICATION':
        categoricalLevels = [int(n) for n in list(
            ee.Dictionary(fullSetPointLocations.aggregate_histogram(classProperty)).keys().getInfo())]
        print('Categorical levels are\n')
        print(categoricalLevels)
        # pyrPolicy = 'mode'
        accuracyMetricString = 'OverallAccuracy'
        # print("The accuracy type used for crossvalidation will be 'overall accuracy'.")
    if modelType == 'PROBABILITY':
        categoricalLevels = [int(n) for n in list(
            ee.Dictionary(fullSetPointLocations.aggregate_histogram(classProperty)).keys().getInfo())]
        print('Categorical levels are\n')
        print(categoricalLevels)
        # pyrPolicy = 'mode'
        accuracyMetricString = 'OverallAccuracy'
        # print("The accuracy type used for crossvalidation will be 'overall accuracy'.")
    else:
        # print('No need to compute categorical levels!')
        # print("The pyramiding policy will be 'mean'.")
        # pyrPolicy = 'mean'
        accuracyMetricString = 'R2'
        categoricalLevels = None

    # Define the asset of CV results
    if mode == 'simple':
        CVResultsAssetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Simple/' + cvAccuracyFCName
    else:
        CVResultsAssetID = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Bootstrapped/' + cvAccuracyFCName
    # Determine the best model
    bestModelName = bestModelNameCV(CVResultsAssetID, accuracyMetricString)
    # remove the classification property name
    bestModelName = bestModelName.replace(classProperty + '_', '')

    # Load the best model using the bestModelName
    if (modelType == 'CLASSIFICATION'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('CLASSIFICATION')
    elif (modelType == 'PROBABILITY'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('PROBABILITY')
    elif (modelType == 'REGRESSION'):
        classifier = ee.Classifier(ee.Feature(
            ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get(
            'c')).setOutputMode('REGRESSION')

    return (bestModelName, classifier, fullSetPointLocations, categoricalLevels)


# simpleMapping(): will create a map using a model with the FULL set of point locations (i.e. not bootstrapped
# or stratified)
# INPUT:
#        fullSetPointLocations = full set of points to run the prediction on
#        classifier = the classifier to use (model) for the prediction
#        pyrPolicy = the pyramiding policy 'mean' for regression and 'mode' for classification
# OUTPUT:
#        the bootstrapped final map of the global predictions as a tiff file.
#        feature importance table with values and plot
def simpleMapping(corrMatrix, fullSetPointLocations, classifier, pyrPolicy, classProperty, composite, pipelineChoice,
                  localROIAssetID=None, noOfChunks=None):
    # Load the composite with the chosen covariates
    updatedComposite = ee.Image(composite)
    exportGeometry = unboundedGeometry
    compositeToClassify = ee.Image(updatedComposite).select(covariateList)

    # Train the classifier with the collection
    trainedClassiferForSingleMap = classifier.train(fullSetPointLocations, classProperty, covariateList)

    # Get the feature importance from the trained classifier and print
    # them to a .csv file and as a bar plot as .png file
    classifierDict = trainedClassiferForSingleMap.explain().get('importance')
    featureImportances = classifierDict.getInfo()
    featureImportances = pd.DataFrame(featureImportances.items(),
                                      columns=['Covariates', 'Feature_Importance']).sort_values(
        by='Feature_Importance',
        ascending=False)
    featureImportances.to_csv(outputPath + '/outputs/mapping/featureImportances' + classProperty + '.csv')
    print('Feature Importances: ', '\n', featureImportances)
    plt = featureImportances.plot(x='Covariates', y='Feature_Importance', kind='bar', legend=False,
                                  title='Feature Importances')
    fig = plt.get_figure()
    fig.savefig(outputPath + '/outputs/mapping/simpleMappingFeatureImportances' + '_' + classProperty + '.png',
                bbox_inches='tight')

    # Classify the image
    classifiedImageSingleMap = compositeToClassify.classify(trainedClassiferForSingleMap,
                                                            classProperty + '_Predicted')

    # Export the image to the assets
    fullSingleImageAsset = ee.batch.Export.image.toAsset(
        image=classifiedImageSingleMap,
        description=classProperty + '_Classified_MapAsset',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/model/' + classProperty + '_SimpleMap',
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=exportGeometry,
        maxPixels=int(1e13),
        pyramidingPolicy={".default": pyrPolicy})
    fullSingleImageAsset.start()

    # Export the image to Cloud Storage.
    fullSingleImageExportCloud = ee.batch.Export.image.toCloudStorage(
        image=classifiedImageSingleMap,
        description=classProperty + '_Classified_Map',
        bucket=bucketOfInterest,
        fileNamePrefix=projectFolder + '/simpleModel/' + classProperty + '_FullSingle_Classified',
        scale=1000,
        region=exportGeometry,
        maxPixels=1e13)
    fullSingleImageExportCloud.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if classProperty in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)
    print('Map created!')
    print('The asset ID for the map is:')
    print('projects/' + usernameFolderString + '/' + projectFolder + '/' + classProperty + '_SimpleMap')

    classifiedMapAsset = ee.Image(
        'projects/' + usernameFolderString + '/' + projectFolder + '/model/' + classProperty + '_SimpleMap')

    # From the classified image extract the observed vs. predicted values
    def extractValues(feat):
        geom = feat.geometry()
        values = classifiedMapAsset.reduceRegion(reducer=ee.Reducer.first(), geometry=geom,
                                                 scale=ee.Image(composite).projection().nominalScale().getInfo(),
                                                 tileScale=16)
        return feat.set(values)

    observedVsPredictedData = fullSetPointLocations.map(extractValues)

    # Export these points
    observedVsPredictedDataExport = ee.batch.Export.table.toCloudStorage(
        collection=observedVsPredictedData,
        description='observedVsPredictedDataExport',
        bucket=bucketOfInterest,
        fileNamePrefix=projectFolder + '/simpleModel/' + classProperty + '_simpleObservedVsPredicted',
        fileFormat='CSV',
        selectors=['Pixel_Lat', 'Pixel_Long', classProperty, classProperty + '_Predicted']
    )
    observedVsPredictedDataExport.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if 'observed' in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)

    # Download all data
    bucket = formattedBucketOI + '/' + projectFolder + '/simpleModel/'
    directory = outputPath + '/outputs/mapping'
    downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
    subprocess.run(downloadBucket)

    # !! Break and wait
    while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
        print('Download to local folder in progress...! ',
              datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(normalWaitTime)
    print('Map images can be found in the local folder.')


def bootstrapSampling(strataDict, corrMatrix):
    # Step 1: Bootstrapping and upload
    # Create bootstrapped samples from the original data (correlation Matrix)
    # and upload them to the Google Cloud Storage Bucket
    # corrMatrix = pd.read_csv(outputPath + '/outputs/dataPrep/finalCorrMatrix.csv')
    # Initiate an empty list to store all the file names that will be uploaded
    fileNameList = []

    # loop through the seeds list (number of bootstrap samples to be created)
    for n in seedsToUseForBootstrapping:
        # Subsample the correlation matrix stratified by the chosen stratification
        # variable (either 'Resolve_Biome' or 'Resolve_Ecoregion')
        # The sample will be of the specified size bootstrapModelSize
        stratSample = corrMatrix.groupby(stratVariable, group_keys=False).apply(
            lambda x: x.sample(n=int(round((strataDict.get(x.name) / 100) * bootstrapModelSize)),
                               replace=True, random_state=n))
        stratSample = duplicateCoordsCol(stratSample, 'Pixel_Lat', 'Pixel_Long')

        # Upload folder for GCSB
        uploadBucket = formattedBucketOI + '/' + projectFolder + '/Upload/'

        # define the output folder
        holdingFolder = outputPath + '/outputs/bootstrapping'

        # Format the title of the CSV and export it to a local directory
        fileNameHeader = 'BootstrapSample_'
        titleOfBootstrapCSV = fileNameHeader + str(n).zfill(3)
        fileNameList.append(titleOfBootstrapCSV)
        fullLocalPath = holdingFolder + '/' + titleOfBootstrapCSV + '.csv'
        stratSample.to_csv(holdingFolder + '/' + titleOfBootstrapCSV + '.csv', index=False)

        # Format the bash call to upload the files to the Google Cloud Storage bucket
        gsutilBashUploadList = [bashFunctionGSUtil] + arglist_preGSUtilUploadFile + [fullLocalPath] + [uploadBucket]
        subprocess.run(gsutilBashUploadList)
        # print(titleOfBootstrapCSV + ' uploaded to a GCSB!')

    # Wait for the GSUTIL uploading process to finish before moving on
    while not all(
            x in subprocess.run([bashFunctionGSUtil, 'ls', uploadBucket], stdout=subprocess.PIPE).stdout.decode('utf-8')
            for x in fileNameList):
        print('Bootstrapping samples are being uploaded to GCSB...')
        time.sleep(5)
    print('All bootstrapping samples have been uploaded to GCSB.')

    # Step 2: From GCSB upload to Google Earth Engine into a specific asset folder
    # Loop through the file names and upload each of them to Earth Engine
    # define the columns of your data signifying latitude and longitude
    arglist_postEEUploadTable = ['--x_column', 'geo_Long', '--y_column', 'geo_Lat']
    # Loop through the file names and upload each of them to Earth Engine
    for f in fileNameList:
        assetIDForBootstrapColl = 'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples'
        gsStorageFileLocation = uploadBucket
        earthEngineUploadTableCommands = [bashFunction_EarthEngine] + arglist_preEEUploadTable + \
                                         [assetIDStringPrefix + assetIDForBootstrapColl + '/' + f] + \
                                         [gsStorageFileLocation + f + '.csv'] + arglist_postEEUploadTable
        subprocess.run(earthEngineUploadTableCommands)
        # print(f   + ' EarthEngine Ingestion started!')

    # print('All files are being ingested.')

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if fileNameHeader in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)
    print('Moving on...')


# bootstrappingMapping(): This function produces bootstrapped samples based on the cleaned data (correlation
# matrix), and then trains a classifier on each of samples, classifies these sample points.
# Finally an averaged map is produced by computing the mean and standard deviation values at each location
# using all the bootstrapped images. This final map is then exported into the GCSB.
# INPUT:
#        classifier = the classifier that was shown to be the best to use with this data
#        strataDict = a stratification dictionary giving the down- or upsampling for samples from specific
#                     stratification variables
#        pyrPolicy = the policy by which the image is pyramided in GEE
# OUTPUT:
#        the bootstrapped final map of the global predictions as a tiff file.
def bootstrappingMapping(classProperty, fullSetPointLocations, classifier, pyrPolicy, noOfBootstrapSamples, composite,
                         pipelineChoice, categories, localROIAssetID=None):
    updatedComposite = ee.Image(composite)
    exportGeometry = unboundedGeometry

    # compute the scale to use for exports
    scaleToUse = ee.Image(updatedComposite).projection().nominalScale().getInfo()

    # for the moment keep this version! Max 100 bootstrap samples!!
    # after I have checked if increaseing size or number of bootstraps increases the model
    # think about adding a workflow that runs the pipeline on chunks of bootstrap samples
    # and averages the results
    if noOfBootstrapSamples > 100:
        noOfBootstrapSamples = 100

    # Train the classifier with the collection
    trainedClassiferForSingleMap = classifier.train(fullSetPointLocations, classProperty, covariateList)

    # Get the feature importance from the trained classifier and print
    # them to a .csv file and as a bar plot as .png file
    classifierDict = trainedClassiferForSingleMap.explain().get('importance')
    featureImportances = classifierDict.getInfo()
    featureImportances = pd.DataFrame(featureImportances.items(),
                                      columns=['Covariates', 'Feature_Importance']).sort_values(
        by='Feature_Importance',
        ascending=False)
    featureImportances.to_csv(outputPath + '/outputs/mapping/featureImportances' + classProperty + '.csv')
    print('Feature Importances: ', '\n', featureImportances)
    plt = featureImportances.plot(x='Covariates', y='Feature_Importance', kind='bar', legend=False,
                                  title='Feature Importances')
    fig = plt.get_figure()
    fig.savefig(outputPath + '/outputs/mapping/simpleMappingFeatureImportances' + '_' + classProperty + '.png',
                bbox_inches='tight')

    # Step 1: Train the model on every bootstrapping sample and predict the values for each
    # point in the bootstrapped samples

    # load the bootstrap samples into ONE image collection and set the values as the
    # training collection
    collectionPath = 'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples/BootstrapSample_'

    # Define a function that sets the sample values as the TrainingColl in each
    # image of the resulting image collection
    def setTrainingColl(seedToUse):
        return ee.Image(0).set('TrainingColl', ee.FeatureCollection(collectionPath + pad(seedToUse, 3)))

    # Map the setTrainingColl over all bootstrapped samples
    ICToMap = ee.ImageCollection(list(map(setTrainingColl, seedsToUseForBootstrapping)))

    # Load the composite with the chosen covariates
    compositeToClassify = ee.Image(updatedComposite).select(covariateList)

    # Define a function that trains the classifier on the training collection
    # meaning on every bootstrapped sample and classifies it on the same samples
    def modelTrainClassify(i):
        # Load the feature collection with training data
        trainingColl = ee.FeatureCollection(i.get('TrainingColl'))

        # Train the classifier on the training data
        trainedBootstrapClassifier = classifier.train(
            features=trainingColl,
            classProperty=classProperty,
            inputProperties=covariateList)

        # Apply the classifier to the composite to make the final map
        bootstrapImage = compositeToClassify.classify(trainedBootstrapClassifier, classProperty + '_predicted')
        return bootstrapImage

    # Map the modelTrainClassify() function over the IC containing all the bootstrapped images
    imageCollectionToReduce = ICToMap.map(modelTrainClassify)

    # Once the bootstrap iterations are complete, run the upper and lower confidence interval
    # bounds (assuming a non-parametric bootstrap)
    # In other words: As a last step we average over all the images to obtain
    # one map showing the mean value and the standard deviation at each point location
    # computed from all the bootstrapped images
    if modelType == 'PROBABILITY':

        # Reduce bootstrap images to mean
        meanImage = imageCollectionToReduce.reduce(reducer=ee.Reducer.mean()).rename('Bootstrapped_Mean')
        print('mean img: ')

        # Reduce bootstrap images to lower and upper CIs
        upperLowerCIImage = imageCollectionToReduce.reduce(
            reducer=ee.Reducer.percentile([2.5, 97.5], ['lower', 'upper']))  # .rename('Bootstrapped_CI')

        # Reduce bootstrap images to standard deviation
        stdDevImage = imageCollectionToReduce.reduce(reducer=ee.Reducer.stdDev()).rename('Bootstrapped_stdDev')

        # Coefficient of Variation: stdDev divided by mean
        coefOfVarImage = stdDevImage.divide(meanImage).rename('Bootstrapped_CoefOfVar')

        finalImageToExport = ee.Image.cat(
            meanImage.toFloat(),
            stdDevImage.toFloat(),
            upperLowerCIImage.toFloat(),
            coefOfVarImage.toFloat())

        # Step 4: Export the final map and data
        # Export final map to Assets and GCSB
        # Export the image to the assets
        fullSingleImageAsset = ee.batch.Export.image.toAsset(
            image=finalImageToExport,
            description=classProperty + '_BootstrappedClassified_MapAsset',
            assetId='projects/' + usernameFolderString + '/' + projectFolder + '/model/BootstrappedImage_Classified' + classProperty,
            crs='EPSG:4326',
            crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
            region=exportGeometry,
            maxPixels=int(1e13),
            pyramidingPolicy={".default": pyrPolicy})
        fullSingleImageAsset.start()

        # export to Cloud storage
        export = ee.batch.Export.image.toCloudStorage(
            image=finalImageToExport,
            description=classProperty + '_BootstrappedClassified_MapAsset',
            bucket=bucketOfInterest,
            fileNamePrefix=projectFolder + '/bootstrappedModel/BootstrappedImage_Classified_' + classProperty + 'Map',
            region=exportGeometry,
            scale=1000,
            maxPixels=1e13
        )
        export.start()

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if 'Bootstrapped' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)

        # Extract observed vs. predicted values from the classified image
        classifiedImageAsset = ee.Image(
            'projects/' + usernameFolderString + '/' + projectFolder + '/model/BootstrappedImage_Classified' + classProperty)

        def bootstrappedObsVsPredExtraction(id):
            # Load the feature collection with training data
            trainingColl = ee.FeatureCollection(id)

            # Extract the observed vs. predicted values for the bootstrapped sample from the exported map asset
            def extractValues(feat):
                geom = feat.geometry()
                values = classifiedImageAsset.reduceRegion(reducer=ee.Reducer.first().forEachBand(classifiedImageAsset),
                                                           geometry=geom, scale=scaleToUse, tileScale=16)
                return feat.set(values)

            # Extract the values
            observedVsPredictedData = trainingColl.map(extractValues)

            # Export these points
            observedVsPredictedDataExport = ee.batch.Export.table.toCloudStorage(
                collection=observedVsPredictedData,
                description='observedVsPredictedDataExport_' + id[-3:],
                bucket=bucketOfInterest,
                fileNamePrefix=projectFolder + '/bootstrappedModel/bootstrappedObservedVsPredicted/bootstrappedObservedVsPredicted_' + classProperty + '_' + id[
                                                                                                                                                             -3:],
                fileFormat='CSV',
                selectors=['Pixel_Lat', 'Pixel_Long', classProperty, 'Bootstrapped_Mean', 'Bootstrapped_stdDev',
                           'instat_class_predicted_lower', 'instat_class_predicted_upper', 'Bootstrapped_CoefOfVar']
            )
            observedVsPredictedDataExport.start()

        # Get the predicted vs observed values for each bootstrapped sample
        list(map(lambda assetId: bootstrappedObsVsPredExtraction(assetId),
                 list([fc['id'] for fc in ICToMap.aggregate_array('TrainingColl').getInfo()])))

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if 'observed' in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
                  count)
            time.sleep(normalWaitTime)

    bucket = formattedBucketOI + '/' + projectFolder + '/bootstrappedModel/'
    directory = outputPath + '/outputs/mapping'
    downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
    subprocess.run(downloadBucket)

    # !! Break and wait
    while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
        print('Download to local folder in progress...! ',
              datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(normalWaitTime)

    print('Map images can be found in the local folder.')


# ------------------------------------------------------------------------------------------------------------- #
# D Evaluation:  Main functions
# ------------------------------------------------------------------------------------------------------------- #

# performanceEvaluation(): This function evaluates the performance of the mapping procedure
# INPUT: the input data set
# OUTPUT: none

def performanceEvaluation(finalData):
    corrMatrix_data = finalData[['Pixel_Lat', 'Pixel_Long', 'Resolve_Biome']]

    obsVsPred_path = outputPath + '/outputs/mapping/bootstrappedModel/bootstrappedObservedVsPredicted'
    obsVsPred_files = glob.glob(obsVsPred_path + '/*')

    # Compute AUC and other performance criteria etc.
    plt.figure(0).clf()
    performanceDF = pd.DataFrame()
    summaryTables = []
    classificationReport = pd.DataFrame()
    for bootFile in sorted(obsVsPred_files):
        obsVsPred_data = pd.read_csv(bootFile)
        obsVsPred_data = obsVsPred_data.dropna()
        obsVsPred_data = obsVsPred_data.merge(corrMatrix_data, how='left', left_on=['Pixel_Lat', 'Pixel_Long'],
                                              right_on=['Pixel_Lat', 'Pixel_Long'], suffixes=('', '_y'))

        # ROC curve for finding the optimal threshold
        # calculate G-mean
        fpr, tpr, threshold = roc_curve(obsVsPred_data.loc[:, classPropList[0]],
                                        obsVsPred_data.loc[:, 'Bootstrapped_Mean'])
        auc = metrics.auc(fpr, tpr)
        df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': threshold})
        gmean = np.sqrt(tpr * (1 - fpr))
        # find optimal threshold
        index = np.argmax(gmean)
        thresholdOpt = round(threshold[index], ndigits=4)
        gmeanOpt = round(gmean[index], ndigits=4)
        fprOpt = round(fpr[index], ndigits=4)
        tprOpt = round(tpr[index], ndigits=4)

        # fig, ax = plt.subplots(1, 1)
        plt.plot(fpr, tpr, c='black', alpha=0.3)
        plt.scatter(x=fprOpt, y=tprOpt, c='black', alpha=0.3)
        # plt.annotate('Optimal threshold \n for class: {}'.format(thresholdOpt), (fprOpt, tprOpt))

        ## Use the threshold, get the class labels and from that the number of invaded vs noninvaded plots
        obsVsPred_data['predicted_class'] = obsVsPred_data['Bootstrapped_Mean'].apply(
            lambda x: 1 if x >= thresholdOpt else 0)
        # print(obsVsPred_data['predicted_class'].value_counts())
        classificationDict = classification_report(obsVsPred_data.loc[:, classPropList[0]],
                                                   obsVsPred_data.loc[:, 'predicted_class'], output_dict=True)
        classificationDF = pd.DataFrame([classificationDict['0'], classificationDict['1']],
                                        columns=['precision', 'recall', 'f1-score', 'support'])
        classificationReport = pd.concat([classificationReport, classificationDF])
        # print(classification_report(obsVsPred_data.loc[:,classPropList[0]], obsVsPred_data.loc[:,'predicted_class']))
        fpr, tpr, threshold = roc_curve(obsVsPred_data.loc[:, classPropList[0]],
                                        obsVsPred_data.loc[:, 'predicted_class'])
        auc = metrics.auc(fpr, tpr)
        df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': threshold})
        # print('auc score: ', auc)
        optDF = pd.DataFrame([[thresholdOpt, gmeanOpt, fprOpt, tprOpt, auc]],
                             columns=['Best Threshold', 'G-Mean', 'FPR', 'TPR', 'AUC'])
        performanceDF = pd.concat([performanceDF, optDF])

        Resolve_biome_dict = {
            'Boreal Forests/Taiga': 11.0,
            'Deserts & Xeric Shrublands': 13.0,
            'Flooded Grasslands & Savannas': 9.0,
            'Mangroves': 14.0,
            'Mediterranean Forests, Woodlands & Scrub': 12.0,
            'Montane Grasslands & Shrublands': 10.0,
            'Temperate Broadleaf & Mixed Forests': 4.0,
            'Temperate Conifer Forests': 5.0,
            'Temperate Grasslands, Savannas & Shrublands': 8.0,
            'Tropical & Subtropical Coniferous Forests': 3.0,
            'Tropical & Subtropical Dry Broadleaf Forests': 2.0,
            'Tropical & Subtropical Grasslands, Savannas & Shrublands': 7.0,
            'Tropical & Subtropical Moist Broadleaf Forests': 1.0,
            'Tundra': 6.0}

        biome_dict_rev = {y: x for x, y in Resolve_biome_dict.items()}

        summaryTable = obsVsPred_data.groupby(['Resolve_Biome', 'predicted_class']).size().reset_index()
        summaryTable.columns = ['Resolve_Biome_Num', 'predicted_class', 'counts']
        summaryTable['Resolve_Biome_Name'] = summaryTable['Resolve_Biome_Num'].apply(lambda x: biome_dict_rev.get(x))
        summaryTable2 = summaryTable.pivot(index='Resolve_Biome_Name', columns='predicted_class',
                                           values='counts').fillna(0)
        summaryTable2['Invasion (%)'] = summaryTable2[summaryTable2.columns[1]].divide(
            summaryTable2[summaryTable2.columns[0]].add(summaryTable2[summaryTable2.columns[1]]))
        summaryTables.append(summaryTable2)

    plt.title('ROC curve of all bootstrapped samples')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.savefig(outputPath + '/outputs/ROC_Curve.pdf')
    plt.show()

    thresholdOpt_mean = round(performanceDF['Best Threshold'].mean(), ndigits=4)
    fprOpt_mean = round(performanceDF['FPR'].mean(), ndigits=4)
    tprOpt_mean = round(performanceDF['TPR'].mean(), ndigits=4)
    auc_mean = round(performanceDF['AUC'].mean(), ndigits=4)

    print('Mean of best threshold: {}'.format(thresholdOpt_mean))
    print('mean FPR: {}, mean TPR: {}'.format(fprOpt_mean, tprOpt_mean))
    print('mean AUC score: ', auc_mean)

    classificationReport = classificationReport.groupby(classificationReport.index).mean()
    classificationReport.to_csv(outputPath + '/outputs/classificationReport.csv')
    print(classificationReport)

    summaryExport = pd.concat(summaryTables)
    summaryExport = summaryExport.groupby(summaryExport.index).mean().multiply(100)
    summaryExport.to_csv(outputPath + '/outputs/summary_invasionPerBiome.csv')
    print(summaryExport)

    # Calibrate the probability scores using Platt Scaling
    obsVsPred_data = pd.concat([pd.read_csv(bootFile).dropna() for bootFile in sorted(obsVsPred_files)])

    # Compute calibration curve
    y_true = obsVsPred_data['instat_class']
    y_pred = obsVsPred_data['Bootstrapped_Mean']
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
    plt.scatter(prob_true, prob_pred, c='black', alpha=0.5)
    # Calculate equation for trendline
    z = np.polyfit(prob_true, prob_pred, 1)
    p = np.poly1d(z)
    plt.plot(prob_true, p(prob_true), c='black', alpha=0.5)

    # Apply Platt Scaling
    lr = LogisticRegression(random_state=0).fit(y_pred.values.reshape(-1, 1), y_true)
    y_cal = lr.predict_proba(y_pred.values.reshape(-1, 1))[:, 1]
    prob_true, prob_cal = calibration_curve(y_true, y_cal, n_bins=20)
    plt.scatter(prob_true, prob_cal, c='blue', alpha=0.5)
    # Calculate equation for trendline
    z = np.polyfit(prob_true, prob_cal, 1)
    p = np.poly1d(z)
    plt.plot(prob_true, p(prob_true), c='blue', alpha=0.5)
    plt.plot([0, 1], [0, 1], c='black', linestyle="--")
    plt.xlabel('True probability')
    plt.ylabel('Predicted probability')
    plt.title('Calibration curve')
    plt.savefig(outputPath + '/outputs/Calibration_Curve.pdf')
    plt.show()

    from sklearn.metrics import log_loss
    log_loss(y_true, y_pred)
    log_loss(y_true, y_cal)
    from sklearn.metrics import brier_score_loss
    (1 - brier_score_loss(y_true, y_cal) / brier_score_loss(y_true, y_pred) )* 100






    print('Platt Scaling intercept: ', np.round(lr.intercept_[0], decimals=4))
    print('Platt Scaling coefficient: ', np.round(lr.coef_[0][0], decimals=4))

    coeffs = pd.DataFrame([[lr.intercept_[0], lr.coef_[0][0]]], columns=['LR Intercept', 'LR Coefficient'])
    coeffs.to_csv(outputPath + '/outputs/plattScaling_coefficients.csv')


def calibrateFinalImage():
    # Calibrate the final image
    classProperty = classPropList[0]
    finalImage = ee.Image(
        'projects/' + usernameFolderString + '/' + projectFolder + '/model/BootstrappedImage_Classified' + classProperty)

    # Load the coefficients
    coeffs = pd.read_csv(outputPath + '/outputs/plattScaling_coefficients.csv')
    b0 = coeffs['LR Intercept']
    b1 = coeffs['LR Coefficient']

    # Prepare a calibration function
    def calibrateImage(image):
        return ee.Image(1) \
            .divide(ee.Image(1) + ee.Image(b0).add(ee.Image(b1).multiply(image)).multiply(ee.Image(-1)).exp())

    # Calibrate all images except for CV (unitless)
    calibratedImage = finalImage.select(['Bootstrapped_Mean', 'Bootstrapped_stdDev', 'instat_class_predicted_lower',
                                         'instat_class_predicted_upper']).addBands(
        finalImage.select(['Bootstrapped_CoefOfVar']))

    # Export the calibrated image
    exportGeometry = unboundedGeometry
    fullSingleImageAsset = ee.batch.Export.image.toAsset(
        image=calibratedImage,
        description=classProperty + '_BootstrappedCalibrated_MapAsset',
        assetId='projects/' + usernameFolderString + '/' + projectFolder + '/model/BootstrappedImage_Calibrated' + classProperty,
        crs='EPSG:4326',
        crsTransform='[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
        region=exportGeometry,
        maxPixels=int(1e13),
        pyramidingPolicy={".default": 'mean'})
    fullSingleImageAsset.start()


# ------------------------------------------------------------------------------------------------------------- #
# Main script
# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# Pipeline:
# ------------------------------------------------------------------------------------------------------------- #
def pipeline(corrMatrix, mode, classPropList, k, modelType, projectFolder, composite, classifierList, pipelineChoice,
             localROIAssetID=None):
    # initiate the pyramiding policy and accuracy metric to be used

    pyrPolicy, accuracyMetricString = pyramidingAccuracy(modelType)

    # compute the stratification dictionary
    strataDict = stratificationDictionary(stratificationChoice, stratVariable, corrMatrix)

    # cross validation:
    subfolderNameCV = 'CV'

    if mode == 'simple':
        subfolderGEE = 'CV_Simple'
    else:
        subfolderGEE = 'CV_Bootstrapped'

    # run the CV for parameter tuning on all classification properties
    for classProp in classPropList:
        CV(mode, corrMatrix, classProp, classifierList, k, accuracyMetricString, subfolderNameCV,
                subfolderGEE)

    # download the CV predictions
    for classProp in classPropList:
        getCvPredictions(mode, modelType, classProp, subfolderGEE)

    # export cv results
    directory = outputPath + '/outputs'
    bucket = formattedBucketOI + '/' + projectFolder + '/' + subfolderGEE
    downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
    subprocess.run(downloadBucket)

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if 'CV' in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:',
              count)
        time.sleep(normalWaitTime)

    # Compute the AUC under the ROC for the CV predictions
    for classProp in classPropList:
        mean_auc, std_auc = cvAUC(classProp)
    print('The CV step is finalized.')
    print('The CV AUC is: %0.2f +- %0.2f)' % (mean_auc, std_auc))

    # modeling:
    if mode == 'simple':

        for classProp in classPropList:
            # preparation
            bestModelName, classifier, fullSetPointLocations, categories = mappingPrep(mode, modelType, classProp)
            print('the best model for the ', classProp, 'is', bestModelName, '\n')
            # mapping procedure
            simpleMapping(corrMatrix, fullSetPointLocations, classifier, pyrPolicy, classProp, composite,
                          pipelineChoice, localROIAssetID, noOfChunks)

        # download the final outputs
        bucket = formattedBucketOI + '/' + projectFolder + '/simpleModel/'
        directory = outputPath + '/outputs/mapping'
        downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
        subprocess.run(downloadBucket)

        # !! Break and wait
        while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
            # print('Download to local folder in progress...! ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            time.sleep(normalWaitTime)
        print('The simple mapping procedure is finalized.')

    else:

        # compute the stratification dictionary
        strataDict = stratificationDictionary(stratificationChoice, stratVariable, corrMatrix)

        # create the bootstrap samples
        bootstrapSampling(strataDict, corrMatrix)

        # Run the mapping on the bootstrap samples
        for classProp in classPropList:
            # preparation
            bestModelName, classifier, fullSetPointLocations, categories = mappingPrep(mode, modelType, classProp)
            print('categories: ', categories)
            bootstrappingMapping(classProp, fullSetPointLocations, classifier, pyrPolicy, noOfBootstrapSamples,
                                 composite, pipelineChoice, categories, localROIAssetID)

        # download the final outputs
        bucket = formattedBucketOI + '/' + projectFolder + '/bootstrappedModel/'
        directory = outputPath + '/outputs/mapping'
        downloadBucket = [bashFunctionGSUtil, '-m', 'cp', '-r', bucket, directory]
        subprocess.run(downloadBucket)
        print('Everything is downloaded.')

        # !! Break and wait
        while any(x in str(ee.batch.Task.list()) for x in ['RUNNING', 'READY']):
            print('Download to local folder in progress...! ',
                  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            time.sleep(normalWaitTime)

        print('The bootstrapped mapping procedure is finalized.')

    message = 'Please check your final outputs in the outputs folder.'
    return print(message)
