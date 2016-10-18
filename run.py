import dl

parameters = {}
parameters['DL']={}
parameters['DS']={}

# set architecture parameters

parameters['DL']['type'] = 'oNetEightpNetEight' # OnetOne, pNetOne, oNetTwo, pNetTwo, oNetpNet, oNetTwopNetTwo, oNetFourpNetFour, oNetEightpNetEight.
parameters['chunkHop'] = 250			# 250 or 80

########################################
# the following parameters were fixed 
# during the different experiments:
########################################
# general parameters
parameters['errorCode'] = 999
parameters['testMethod'] = 'all'                    # options: 'majorityVote' or 'utterances' or 'all'
parameters['mode'] = 'train'     		    # options: 'train' or test. For test, introduce the mode(l) to be loaded. i.e. 'genres_cnn1_201814200862305619998386402939002111166'.
parameters['folds']=10                              # for k-fold cross-validation. if parameters['folds']==1: then the following pre-defined splits apply (meaning: NO cross-validation)
parameters['trainSplit'] = 0.8 
parameters['testSplit'] = 0.1        		    # 1 (using all dataset for testing)
parameters['valSplit'] = 1-parameters['trainSplit']-parameters['testSplit']
parameters['randomTry'] = 1    			    # how many times do you want to try with different random initializations per fold?
parameters['random_seed'] = 11                      # random seed (None will turn off random seed and use a random init)
parameters['use_random_seed'] = True                # use a fixed random seed for network initialization

# Deep Learning parameters
parameters['DL']['num_epochs'] = 250
parameters['DL']['batchSize'] = 50
parameters['DL']['lr'] = 0.01
parameters['DL']['momentum'] = 0
parameters['DL']['cost'] = 'crossentropy'           

# Data Set (input data) parameters
parameters['DS']['dataset'] = 'Ballroom'	    # options: 'Ballroom' or 'genres', for the GTZAN.
parameters['DS']['frameSize'] = 2048
parameters['DS']['hopSize'] = 1024
parameters['DS']['specTransform'] = 'mel'           # options: 'mel' and 'magnitudeSTFT'
parameters['DS']['numChannels'] = 1
parameters['DS']['windowType'] = 'blackmanharris62'
parameters['DS']['yInput'] = parameters['DS']['melBands']
parameters['DS']['inputNormWhere'] = 'global'
parameters['DS']['inputNorm'] = 'log0m1v'           # options: 'log0m1v' or 'None'
parameters['DS']['melBands'] = 40
parameters['DS']['xInput'] = 250   

dl.main(parameters)
