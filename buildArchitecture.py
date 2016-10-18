import lasagne
import numpy as np

def buildNet(input_var, parameters):
    'Select a deep learning architecture'
    
    if parameters['DL']['type']=='oNetOne':
        return oNetOne(input_var, parameters)
    elif parameters['DL']['type']=='pNetOne':
        return pNetOne(input_var, parameters)
    elif parameters['DL']['type']=='oNetTwo':
        return oNetTwo(input_var, parameters)
    elif parameters['DL']['type']=='pNetTwo':
        return pNetTwo(input_var, parameters)
    elif parameters['DL']['type']=='oNetpNet':
        return oNetpNet(input_var, parameters)
    elif parameters['DL']['type']=='oNetTwopNetTwo':
        return oNetTwopNetTwo(input_var, parameters)
    elif parameters['DL']['type']=='oNetFourpNetFour':
        return oNetFourpNetFour(input_var, parameters)
    elif parameters['DL']['type']=='oNetEightpNetEight':
        return oNetEightpNetEight(input_var, parameters)
    else:
        print 'Architecture NOT supported'

#####################

def oNet(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    network={}

    network["input"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)

    network["o41"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,41),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o41"] = lasagne.layers.MaxPool2DLayer(network["o41"], pool_size=(4,network["o41"].output_shape[3]))

    network["o36"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,36),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o36"] = lasagne.layers.MaxPool2DLayer(network["o36"], pool_size=(4,network["o36"].output_shape[3]))

    network["o31"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,31),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o31"] = lasagne.layers.MaxPool2DLayer(network["o31"], pool_size=(4,network["o31"].output_shape[3]))

    network["o26"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,26),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o26"] = lasagne.layers.MaxPool2DLayer(network["o26"], pool_size=(4,network["o26"].output_shape[3]))

    network["o21"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,21),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o21"] = lasagne.layers.MaxPool2DLayer(network["o21"], pool_size=(4,network["o21"].output_shape[3]))

    network["o16"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,16),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o16"] = lasagne.layers.MaxPool2DLayer(network["o16"], pool_size=(4,network["o16"].output_shape[3]))

    network["o11"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,11),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o11"] = lasagne.layers.MaxPool2DLayer(network["o11"], pool_size=(4,network["o11"].output_shape[3]))

    network["o6"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=5, filter_size=(1,6),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_o6"] = lasagne.layers.MaxPool2DLayer(network["o6"], pool_size=(4,network["o6"].output_shape[3]))

    network["optOnset_array"] = lasagne.layers.ConcatLayer([network["pool_o6"], network["pool_o11"],network["pool_o16"],network["pool_o21"],network["pool_o26"],network["pool_o31"],network["pool_o36"],network["pool_o41"]], axis=1, cropping=None)

    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["optOnset_array"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["optOnset_array"],network,parameters

#####################

def pNet(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    network={}

    network["input"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)

    network["t216"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,216),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t216"] = lasagne.layers.MaxPool2DLayer(network["t216"], pool_size=(4,network["t216"].output_shape[3]))

    network["t211"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,211),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t211"] = lasagne.layers.MaxPool2DLayer(network["t211"], pool_size=(4,network["t211"].output_shape[3]))

    network["t206"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,206),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t206"] = lasagne.layers.MaxPool2DLayer(network["t206"], pool_size=(4,network["t206"].output_shape[3]))

    network["t201"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,201),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t201"] = lasagne.layers.MaxPool2DLayer(network["t201"], pool_size=(4,network["t201"].output_shape[3]))

    network["t196"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,196),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t196"] = lasagne.layers.MaxPool2DLayer(network["t196"], pool_size=(4,network["t196"].output_shape[3]))

    network["t191"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,191),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t191"] = lasagne.layers.MaxPool2DLayer(network["t191"], pool_size=(4,network["t191"].output_shape[3]))

    network["t186"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,186),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t186"] = lasagne.layers.MaxPool2DLayer(network["t186"], pool_size=(4,network["t186"].output_shape[3]))

    network["t181"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,181),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t181"] = lasagne.layers.MaxPool2DLayer(network["t181"], pool_size=(4,network["t181"].output_shape[3]))

    network["t176"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,176),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t176"] = lasagne.layers.MaxPool2DLayer(network["t176"], pool_size=(4,network["t176"].output_shape[3]))

    network["t171"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,171),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t171"] = lasagne.layers.MaxPool2DLayer(network["t171"], pool_size=(4,network["t171"].output_shape[3]))

    network["t166"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,166),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t166"] = lasagne.layers.MaxPool2DLayer(network["t166"], pool_size=(4,network["t166"].output_shape[3]))

    network["t161"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,161),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t161"] = lasagne.layers.MaxPool2DLayer(network["t161"], pool_size=(4,network["t161"].output_shape[3]))

    network["t156"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,156),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t156"] = lasagne.layers.MaxPool2DLayer(network["t156"], pool_size=(4,network["t156"].output_shape[3]))

    network["t151"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,151),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t151"] = lasagne.layers.MaxPool2DLayer(network["t151"], pool_size=(4,network["t151"].output_shape[3]))

    network["t146"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,146),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t146"] = lasagne.layers.MaxPool2DLayer(network["t146"], pool_size=(4,network["t146"].output_shape[3]))

    network["t141"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,141),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t141"] = lasagne.layers.MaxPool2DLayer(network["t141"], pool_size=(4,network["t141"].output_shape[3]))

    network["t136"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,136),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t136"] = lasagne.layers.MaxPool2DLayer(network["t136"], pool_size=(4,network["t136"].output_shape[3]))

    network["t131"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,131),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t131"] = lasagne.layers.MaxPool2DLayer(network["t131"], pool_size=(4,network["t131"].output_shape[3]))

    network["t126"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,126),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t126"] = lasagne.layers.MaxPool2DLayer(network["t126"], pool_size=(4,network["t126"].output_shape[3]))

    network["t121"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,121),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t121"] = lasagne.layers.MaxPool2DLayer(network["t121"], pool_size=(4,network["t121"].output_shape[3]))

    network["t116"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,116),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t116"] = lasagne.layers.MaxPool2DLayer(network["t116"], pool_size=(4,network["t116"].output_shape[3]))

    network["t111"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,111),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t111"] = lasagne.layers.MaxPool2DLayer(network["t111"], pool_size=(4,network["t111"].output_shape[3]))

    network["t106"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,106),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t106"] = lasagne.layers.MaxPool2DLayer(network["t106"], pool_size=(4,network["t106"].output_shape[3]))

    network["t101"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,101),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t101"] = lasagne.layers.MaxPool2DLayer(network["t101"], pool_size=(4,network["t101"].output_shape[3]))

    network["t96"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,96),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t96"] = lasagne.layers.MaxPool2DLayer(network["t96"], pool_size=(4,network["t96"].output_shape[3]))

    network["t91"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,91),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t91"] = lasagne.layers.MaxPool2DLayer(network["t91"], pool_size=(4,network["t91"].output_shape[3]))

    network["t86"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,86),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t86"] = lasagne.layers.MaxPool2DLayer(network["t86"], pool_size=(4,network["t86"].output_shape[3]))

    network["t81"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,81),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t81"] = lasagne.layers.MaxPool2DLayer(network["t81"], pool_size=(4,network["t81"].output_shape[3]))

    network["t76"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,76),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t76"] = lasagne.layers.MaxPool2DLayer(network["t76"], pool_size=(4,network["t76"].output_shape[3]))

    network["t71"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,71),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t71"] = lasagne.layers.MaxPool2DLayer(network["t71"], pool_size=(4,network["t71"].output_shape[3]))

    network["t66"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,66),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t66"] = lasagne.layers.MaxPool2DLayer(network["t66"], pool_size=(4,network["t66"].output_shape[3]))

    network["t61"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,61),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t61"] = lasagne.layers.MaxPool2DLayer(network["t61"], pool_size=(4,network["t61"].output_shape[3]))

    network["t56"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,56),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t56"] = lasagne.layers.MaxPool2DLayer(network["t56"], pool_size=(4,network["t56"].output_shape[3]))

    network["t51"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,51),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t51"] = lasagne.layers.MaxPool2DLayer(network["t51"], pool_size=(4,network["t51"].output_shape[3]))

    network["t46"] = lasagne.layers.Conv2DLayer(network["input"], num_filters=1, filter_size=(1,46),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    network["pool_t46"] = lasagne.layers.MaxPool2DLayer(network["t46"], pool_size=(4,network["t46"].output_shape[3]))

    network["optTime_array"] = lasagne.layers.ConcatLayer([network["pool_t46"], \
                                                           network["pool_t51"],network["pool_t56"], \
                                                           network["pool_t61"],network["pool_t66"], \
                                                           network["pool_t71"],network["pool_t76"], \
                                                           network["pool_t81"],network["pool_t86"], \
                                                           network["pool_t91"],network["pool_t96"], \
                                                           network["pool_t101"],network["pool_t106"], \
                                                           network["pool_t111"],network["pool_t116"], \
                                                           network["pool_t121"],network["pool_t126"], \
                                                           network["pool_t131"],network["pool_t136"], \
                                                           network["pool_t141"],network["pool_t146"], \
                                                           network["pool_t151"],network["pool_t156"], \
                                                           network["pool_t161"],network["pool_t166"], \
                                                           network["pool_t171"],network["pool_t176"], \
                                                           network["pool_t181"],network["pool_t186"], \
                                                           network["pool_t191"],network["pool_t196"], \
                                                           network["pool_t201"],network["pool_t206"], \
                                                           network["pool_t211"],network["pool_t216"]], axis=1, cropping=None)

    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["optTime_array"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["optTime_array"],network,parameters

#####################

def oNetOne(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet,o_oNet_layers,parameters=oNet(input_var, parameters)

    network={}

    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(o_oNet, p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters


#####################

def pNetOne(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_pNet,o_pNet_layers,parameters=pNet(input_var, parameters)

    network={}

    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(o_pNet, p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def oNetTwo(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet,o_oNet_layers,parameters=oNet(input_var, parameters)
    o_oNet2,o_oNet2_layers,parameters=oNet(input_var, parameters)

    network={}

    network["fusion"] = lasagne.layers.ConcatLayer([o_oNet, o_oNet2], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def pNetTwo(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_pNet,o_pNet_layers,parameters=pNet(input_var, parameters)
    o_pNet1,o_pNet_layers1,parameters=pNet(input_var, parameters)

    network={}
    network["fusion"] = lasagne.layers.ConcatLayer([o_pNet, o_pNet1], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def oNetpNet(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet,o_oNet_layers,parameters=oNet(input_var, parameters)
    o_pNet,o_pNet_layers,parameters=pNet(input_var, parameters)

    network={}

    network["fusion"] = lasagne.layers.ConcatLayer([o_oNet, o_pNet], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def oNetTwopNetTwo(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet,o_oNet_layers,parameters=oNet(input_var, parameters)
    o_oNet2,o_oNet_layers2,parameters=oNet(input_var, parameters)
    o_pNet,o_pNet_layers,parameters=pNet(input_var, parameters)
    o_pNet2,o_pNet_layers2,parameters=pNet(input_var, parameters)

    network={}

    network["fusion"] = lasagne.layers.ConcatLayer([o_oNet, o_pNet, o_oNet2, o_pNet2], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def oNetFourpNetFour(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet,o_oNet_layers,parameters=oNet(input_var, parameters)
    o_oNet2,o_oNet_layers2,parameters=oNet(input_var, parameters)
    o_oNet3,o_oNet_layers3,parameters=oNet(input_var, parameters)
    o_oNet4,o_oNet_layers4,parameters=oNet(input_var, parameters)
    o_pNet,o_pNet_layers,parameters=pNet(input_var, parameters)
    o_pNet2,o_pNet_layers2,parameters=pNet(input_var, parameters)
    o_pNet3,o_pNet_layers3,parameters=pNet(input_var, parameters)
    o_pNet4,o_pNet_layers4,parameters=pNet(input_var, parameters)

    network={}

    network["fusion"] = lasagne.layers.ConcatLayer([o_oNet, o_pNet, o_oNet2, o_pNet2, o_oNet3, o_pNet3, o_oNet4, o_pNet4], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters

#####################

def oNetEightpNetEight(input_var, parameters):

    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.HeUniform()

    o_oNet1,o_oNet_layers1,parameters=oNet(input_var, parameters)
    o_oNet2,o_oNet_layers2,parameters=oNet(input_var, parameters)
    o_oNet3,o_oNet_layers3,parameters=oNet(input_var, parameters)
    o_oNet4,o_oNet_layers4,parameters=oNet(input_var, parameters)
    o_oNet5,o_oNet_layers,parameters=oNet(input_var, parameters)
    o_oNet6,o_oNet_layers2,parameters=oNet(input_var, parameters)
    o_oNet7,o_oNet_layers3,parameters=oNet(input_var, parameters)
    o_oNet8,o_oNet_layers4,parameters=oNet(input_var, parameters)
    o_pNet1,o_pNet_layers1,parameters=pNet(input_var, parameters)
    o_pNet2,o_pNet_layers2,parameters=pNet(input_var, parameters)
    o_pNet3,o_pNet_layers3,parameters=pNet(input_var, parameters)
    o_pNet4,o_pNet_layers4,parameters=pNet(input_var, parameters)
    o_pNet5,o_pNet_layers5,parameters=pNet(input_var, parameters)
    o_pNet6,o_pNet_layers6,parameters=pNet(input_var, parameters)
    o_pNet7,o_pNet_layers7,parameters=pNet(input_var, parameters)
    o_pNet8,o_pNet_layers8,parameters=pNet(input_var, parameters)

    network={}

    network["fusion"] = lasagne.layers.ConcatLayer([o_oNet1, o_pNet1, o_oNet2, o_pNet2, o_oNet3, o_pNet3, o_oNet4, o_pNet4, o_oNet5, o_pNet5, o_oNet6, o_pNet6, o_oNet7, o_pNet7, o_oNet8, o_pNet8], axis=1, cropping=None)
    network["output"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["fusion"], p=0.5),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["output"],network,parameters
