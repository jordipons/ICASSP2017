#!/usr/bin/env python

import time, random
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
import pickle

import theano, lasagne
import theano.tensor as T

import loadDatasets as loadData
import buildArchitecture as buildArch

import matplotlib.pyplot as plt

def main(parameters):

    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):          # BE CAREFUL! if the last examples are not enough to create a batch, are descarted.
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    if parameters['use_random_seed']:
        np.random.seed(parameters['random_seed'])

    meanUtterances=[]
    meanMajorityVote=[]
    hashs=[]
    y_true=[]
    y_predicted=[]
    lrOriginal=parameters['DL']['lr']

    for parameters['currentFold'] in range(0, parameters['folds']):
        maxUtterances=[]
        maxMajorityVote=[]
        maxhashs=[]
        for rt in range(0,parameters['randomTry']):
            
            print '## Evaluating fold num: '+str(parameters['currentFold'])
            parameters['DL']['lr']=lrOriginal
            print parameters

            print(" - Loading data..")

            X_train, y_train, X_val, y_val, X_test_utterances   , y_test_utterances, X_test_majorityVote, y_test_majorityVote, parameters = loadData.load_dataset(parameters)

            print(" - Building network..")

            input_var = T.tensor4('inputs')
            target_var = T.ivector('targets')
            network,netLayers,parameters=buildArch.buildNet(input_var,parameters)

            parameters['DL']['numParamArchitecture'] = lasagne.layers.count_params(network)

            print '\t Number of parameters: ' + str(parameters['DL']['numParamArchitecture'])

            print(" - Compiling functions..")

            def computeLoss(prediction, target_var, parameters):
                if parameters['DL']['cost']=='crossentropy':
                    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
                elif parameters['DL']['cost']=='squared_error':
                    loss = lasagne.objectives.squared_error(prediction, target_var)
                loss = loss.mean()
                return loss

            def compileFn():
                # define training functions
                prediction = lasagne.layers.get_output(network)
                train_loss =computeLoss(prediction, target_var, parameters)
                train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
                params = lasagne.layers.get_all_params(network, trainable=True)
                updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=parameters['DL']['lr'], momentum=parameters['DL']['momentum'])

                # define testing/val functions
                test_prediction = lasagne.layers.get_output(network, deterministic=True)
                test_loss=computeLoss(test_prediction, target_var, parameters)
                test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

                # compile training and test/val functions
                train_fn = theano.function([input_var, target_var], [train_loss, train_acc], updates=updates)
                val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
                predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1)) # outputs probabilities

                conv_act = lasagne.layers.get_output(netLayers["output"], deterministic=True)
                conv_activations=theano.function([input_var], conv_act)

                return train_fn,val_fn,predict_fn,conv_activations

            train_fn,val_fn,predict_fn,conv_activations=compileFn()

            if parameters['mode'] == 'train':
                print(" - Training..")

                hash = random.getrandbits(128)
                ansLoss=np.inf
                countRaisingLoss=0
                valAccuracy_ans=0
                #valLoss_ans=np.inf
                for epoch in range(parameters['DL']['num_epochs']):
                    # training set
                    train_err = 0
                    train_acc = 0
                    train_batches = 0
                    start_time = time.time()
                    for batch in iterate_minibatches(X_train, y_train, parameters['DL']['batchSize'], shuffle=True):
                        inputs, targets = batch
                        err1, acc1 = train_fn(inputs, targets)
                        train_err += err1
                        train_acc += acc1
                        train_batches += 1

                    # validation set
                    val_err = 0
                    val_acc = 0
                    val_batches = 0
                    for batch in iterate_minibatches(X_val, y_val, parameters['DL']['batchSize'], shuffle=False):
                        inputs, targets = batch
                        err, acc = val_fn(inputs, targets)
                        val_err += err
                        val_acc += acc
                        val_batches += 1

                    # output
                    print("    Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, parameters['DL']['num_epochs'], time.time() - start_time))
                    print("      training loss:\t\t{:.6f}".format(train_err / train_batches))
                    print("      training accuracy:\t\t{:.2f} %".format(train_acc / train_batches*100))
                    print("      validation loss:\t\t{:.6f}".format(val_err / val_batches))
                    print("      validation accuracy:\t\t{:.2f} %".format(
                        val_acc / val_batches * 100))

                    ### THIS REQUIRES LONGER RUNS TO BE BENEFICIOUS. But seems to work well. number of epochs set to 2000.
                    if train_err / train_batches > ansLoss:
                        countRaisingLoss=countRaisingLoss+1
                        print 'Counter raised: '+str(countRaisingLoss)
                        if countRaisingLoss>40:
                            break
                    else:
                        if countRaisingLoss>20:
                            parameters['DL']['lr']=parameters['DL']['lr']/2
                            print 'Compiling..'
                            train_fn,val_fn,predict_fn,conv_activations=compileFn()
                            print 'Learning rate changed!'
                            countRaisingLoss=0
                        ansLoss=train_err / train_batches ## NO TINC CLAR SI FER-HO AMB INDENT O SENSE.


                    #################  STORING OUTPUTS INTO FILES FOR TRAINING TRACKING ##################
                    name='./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)
                    # save the best model
                    if (val_acc / val_batches * 100)>valAccuracy_ans:
                    #if (val_err / val_batches) < valLoss_ans:
                        np.savez(name, *lasagne.layers.get_all_param_values(network))
                        res = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.result', 'w')
                        res.write("    Epoch {} of {} took {:.3f}s\n".format(epoch + 1, parameters['DL']['num_epochs'], time.time() - start_time))
                        res.write("      training loss:\t\t{:.6f}\n".format(train_err / train_batches))
                        res.write("      training accuracy:\t\t{:.2f} %".format(train_acc / train_batches*100))
                        res.write("      validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
                        res.write("      validation accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches * 100))
                        res.close()
                        valAccuracy_ans=(val_acc / val_batches * 100)
                        #valLoss_ans = val_err / val_batches
                    # save parameters
                    if epoch==0:
                        param = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.param', 'w')
                        for key, value in parameters.iteritems():
                            if type(value)==type({}):
                                for k in value:
                                    param.write('-'+str(k))
                            else:
                                param.write('-'+str(key))
                        param.write('\n')
                        for key, value in parameters.iteritems():
                            if type(value)==type({}):
                                for k in value:
                                    param.write('-'+str(value[k]))
                            else:
                                param.write('-'+str(value))
                        param.write('\n')
                        param.close()
                        tr = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.training', 'w')
                        tr.write('epoch,trainingLoss,trainingAccuracy,validationLoss,validationAccuracy\n')
                        tr.close()
                    # save training evolution
                    tr = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.training', 'a')
                    tr.write(str(epoch)+','+str(train_err/train_batches)+','+str(train_acc / train_batches*100)+','+str(val_err / val_batches)+','+str(val_acc / val_batches * 100)+'\n')
                    tr.close()
                    #########################################################################################

            print(" - Testing with the best model..")

            if parameters['mode'] != 'train':
                name = './data/preloaded/models/'+parameters['mode']
            else:
                print("   "+str(hash))

            # load best model
            with np.load(name+'.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

            if parameters['testMethod'] == 'all' or parameters['testMethod'] == 'utterances':
                X_test = X_test_utterances
                y_test = y_test_utterances

                test_err = 0
                test_acc = 0
                test_batches = 0
                for batch in iterate_minibatches(X_test, y_test, parameters['DL']['batchSize'], shuffle=False):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    test_err += err
                    test_acc += acc
                    test_batches += 1

                resultsUtterances = test_acc / test_batches * 100
                print("   [UTTERANCES] Final results:")
                print("   test loss:\t\t\t{:.6f}".format(test_err / test_batches))
                print("   test accuracy:\t\t{:.2f} %".format(resultsUtterances))

                #################  STORING RESULTS INTO FILES FOR TRACKING ##################
                if parameters['mode'] != 'train':
                    res = open('./data/results/'+parameters['mode']+'.result', 'a')
                else:
                    res = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.result', 'a')
                res.write("\n[UTTERANCES] Final results:\n")
                res.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
                res.write("  test accuracy:\t\t{:.2f} %\n".format(resultsUtterances))
                res.close()
                #############################################################################

            if parameters['testMethod'] == 'all' or parameters['testMethod'] == 'majorityVote':

                X_test = X_test_majorityVote
                y_test = y_test_majorityVote

                test_acc = 0
                num_tests = 0
                count=-1
                target=np.zeros(1,dtype=np.uint8)+parameters['errorCode']
                for X in X_test:
                    count=count+1
                    target[0]=y_test[count]
                    voting=[]
                    actAns=[]
                    for c in loadData.chunk(X,parameters['DS']['xInput'],parameters['chunkHop']):
                        input=c.reshape(1,parameters['DS']['numChannels'],parameters['DS']['yInput'],parameters['DS']['xInput'])
                        voting.append(predict_fn(input)[0])
                    votes = Counter(voting)
                    mostVoted = votes.most_common(1)[0][0]
                    y_true.append(target[0])
                    y_predicted.append(mostVoted)
                    if mostVoted ==target:
                        test_acc = test_acc+1
                    num_tests += 1

                # output
                resultsMajorityVote=float(test_acc) / num_tests * 100
                print("   [MAJORITYVOTE] Final results:")
                print("   test accuracy:\t\t{:.2f} %\n".format(resultsMajorityVote))

                #################  STORING OUTPUTS INTO FILES FOR TRACKING ##################
                if parameters['mode'] != 'train':
                    res = open('./data/results/'+parameters['mode']+'.result', 'a')
                else:
                    res = open('./data/results/'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hash)+'.result', 'a')
                res.write("\n[MAJORITYVOTE] Final results:\n")
                res.write("  test accuracy:\t\t{:.2f} %\n".format(resultsMajorityVote))
                res.close()
                #############################################################################

            maxUtterances.append(resultsUtterances)
            maxMajorityVote.append(resultsMajorityVote)
            maxhashs.append(hash)

        bestModelFoldUtterances = np.max(maxUtterances)
        bestModelFoldMajorityVote = np.max(maxMajorityVote)
        hashs.append(maxhashs[maxMajorityVote.index(np.max(maxMajorityVote))])

        meanUtterances.append(bestModelFoldUtterances)
        meanMajorityVote.append(bestModelFoldMajorityVote)

    results={}
    results['utterances'] = np.mean(meanUtterances)
    results['majorityVote'] = np.mean(meanMajorityVote)

    cm = confusion_matrix(y_true, y_predicted)

    hashCV = random.getrandbits(128)
    with open('./data/results/CrossValidation_'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hashCV)+'.confusionMatrix', 'w') as f:
        pickle.dump([cm], f)

    if parameters['mode'] == 'train':

        print("########################")
        print parameters
        for h in hashs:
            print(str(h))
        print('')
        print("[CV-UTTERANCES] test accuracy:")
        print("  mean:\t\t{:.2f} %".format(np.mean(meanUtterances)))
        print("  std:\t\t{:.2f} %".format(np.std(meanUtterances)))
        print("[CV-MAJORITYVOTE]:")
        print("  mean:\t\t{:.2f} %".format(np.mean(meanMajorityVote)))
        print("  std:\t\t{:.2f} %".format(np.std(meanMajorityVote)))
        print '\t Number of parameters: ' + str(parameters['DL']['numParamArchitecture'])

        #################  STORING OUTPUTS INTO FILES FOR TRACKING ##################
        res = open('./data/results/CrossValidation_'+parameters['DS']['dataset']+'_'+parameters['DL']['type']+'_'+str(hashCV)+'.result', 'a')
        for h in hashs:
            res.write(str(h))
            res.write("\n")
        res.write("[CV-UTTERANCES] test accuracy:\n")
        res.write("  mean:\t\t{:.2f} %\n".format(np.mean(meanUtterances)))
        res.write("  std:\t\t{:.2f} %\n".format(np.std(meanUtterances)))
        res.write("[CV-MAJORITYVOTE]:\n")
        res.write("  mean:\t\t{:.2f} %\n".format(np.mean(meanMajorityVote)))
        res.write("  std:\t\t{:.2f} %\n".format(np.std(meanMajorityVote)))
        res.close()
        #############################################################################

    return results

