# This file is an installation file that runs the necessary installations
# of programs and packages.
# And it prepares the output directory.
#
# You need to set the directories where to install them.
#
# ----------------------------------------------------------------------------------------------------------- #
from configurations.user_config import *

# ------------------------------------------------------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------------------------------------------------------- #
def prepareOutputDirectory(path):
    outputPath = path
    if not os.path.exists(path): os.mkdir(path)
    outputPath = path + '/outputs'
    if not os.path.exists(outputPath): os.mkdir(outputPath)
    # data Prep
    dataPrepPath = outputPath + '/dataPrep'
    if not os.path.exists(dataPrepPath): os.mkdir(dataPrepPath)
    # kFoldCV
    kFoldCVPath = outputPath + '/CV'
    if not os.path.exists(kFoldCVPath): os.mkdir(kFoldCVPath)
    # bootstrapping
    bootstrappingPath = outputPath + '/bootstrapping'
    if not os.path.exists(bootstrappingPath): os.mkdir(bootstrappingPath)
    # mapping
    mappingPath = outputPath + '/mapping'
    if not os.path.exists(mappingPath): os.mkdir(mappingPath)
    return print('The output directory can be found at the specified path.')


def setUpGEE():
    # in your GEE project folder create a new folder
    folderName = 'projects/' + usernameFolderString + '/' + projectFolder
    print(folderName)
    subprocess.run(bashCommandList_CreateFolder + [folderName])

    # create additional subfolders:
    # gapFilling
    subfolderGapFilling = 'projects/' + usernameFolderString + '/' + projectFolder + '/gapFilling'
    subprocess.run(bashCommandList_CreateFolder + [subfolderGapFilling])

    # ext vs. interpolation
    subfolderExtVsInt = 'projects/' + usernameFolderString + '/' + projectFolder + '/extVsInt'
    subprocess.run(bashCommandList_CreateFolder + [subfolderExtVsInt])

    # kFoldCV: for simple and bootstrapped version
    # simple:
    subfolderkFoldCVSimple = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Simple'
    subprocess.run(bashCommandList_CreateFolder + [subfolderkFoldCVSimple])

    # bootstrapped:
    subfolderkFoldCVBootstrapped = 'projects/' + usernameFolderString + '/' + projectFolder + '/CV_Bootstrapped'
    subprocess.run(bashCommandList_CreateFolder + [subfolderkFoldCVBootstrapped])

    # And subsamples kFoldCVSubsamples
    subfolderkFoldCVBootstrappedSubsamples = subfolderkFoldCVBootstrapped + '/CVSubsamples'
    subprocess.run(bashCommandList_CreateFolder + [subfolderkFoldCVBootstrappedSubsamples])

    # bootstrapSamples
    # Initiate the name of a folder used to hold the bootstrap collections in GEE
    subfolderBootstrap = 'projects/' + usernameFolderString + '/' + projectFolder + '/bootstrappedSamples'
    # Create the image collection before classifying each of the bootstrap images
    subprocess.run(bashCommandList_CreateFolder + [subfolderBootstrap])

    # model
    subfolderModel = 'projects/' + usernameFolderString + '/' + projectFolder + '/model'
    subprocess.run(bashCommandList_CreateFolder + [subfolderModel])

    # Wait until folder created
    while any(x in subprocess.run(bashCommandList_Detect + [folderName], stdout=subprocess.PIPE).stdout.decode(
            'utf-8') for x in stringsOfInterest):
        print('Waiting for asset to be created...')
        time.sleep(normalWaitTime)
    print('Folder created!')

# ------------------------------------------------------------------------------------------------------------- #
# Main functions
# ------------------------------------------------------------------------------------------------------------- #

def setUp(path, folderName):
    # set up output folders
    prepareOutputDirectory(path)

    # set up folder on GEE
    setUpGEE()
    return print('Set up is completed.')
