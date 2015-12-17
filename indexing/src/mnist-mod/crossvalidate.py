'''
Implementation of performing cross-validation with a classifier that has a
train method and a predict method, like those in the learners module.
'''

from printTable import printTable

import threading
import numpy as np
import random
import sys
import itertools

def _train_oneiter(classifier, xpieces, ypieces, epochs, batch_size, idx):
    '''
    Runs one iteration of training for the given k and single instance of
    of a classifier.

    @param classifier - Constructed learning algorithm object, ready to train
    @param xpieces - List of broken xdata into k pieces
    @param ypieces - List of broken ydata into k pieces
    @param epochs - Number of epochs to do in training
    @param batch_size - size of batches for mini-batch stocastic gradient
                        descent
    @param idx - Which iteration of the k-fold cross-validation

    @return accuracy for this iteration as a tuple
        (test_accuracy, train_accuracy)
        Each numer is between 0 and 1)
    '''
    xtrain = xpieces[:idx] + xpieces[idx+1:]
    ytrain = ypieces[:idx] + ypieces[idx+1:]
    train = zip(xtrain, ytrain)
    xtest = xpieces[idx]
    ytest = ypieces[idx]
    random.shuffle(train)
    xdata = np.concatenate([x[0] for x in train], axis=0)
    ydata = np.concatenate([x[1] for x in train], axis=0)
    classifier.train(xdata, ydata, epochs, batch_size)
    #for subtrain in train:
    #    classifier.train(subtrain[0], subtrain[1], epochs, batch_size)
    accuracy = 1 - classifier.errors(xtest, ytest)
    train_accuracy = 1 - classifier.errors(xdata, ydata)
    return (accuracy, train_accuracy)

def _crossvalidate_internal(cls, xdata, ydata, k, epochs, batch_size,
                            hyperparams, names, trainfunc):
    '''
    Internal cross-validation method used by the public methods in order to
    share functionality.

    @param cls - Class for the classifier
    @param xdata - features array (2D numpy array)
    @param ydata - labels (1D numpy array)
    @param k - Defines the k-fold cross-validation
    @param epochs - Number of epochs to use in cross-validation
    @param batch_size - batch size for mini-batch stochastic gradient descent
    @param hyperparams - list of tuples of hyper-parameters to pass to cls
    @param names - Names of the hyper-parameters in hyperparams
    @param trainfunc - Function to use in training
        Parameters:
        - cls: same as above
        - xpieces: list of xdata broken into k pieces
        - ypieces: list of ydata broken into k pieces
        - epochs: same as above
        - batch_size: same as above
        - params_k_product: equal to
             itertools.product(hyperparams, range(k))
    '''
    # Shuffle and copy the data before cross-validation
    perm = np.random.permutation(len(xdata))
    xdata[:] = xdata[perm].copy()
    ydata[:] = ydata[perm].copy()
    test_percents = np.zeros(len(hyperparams))
    train_percents = np.zeros(len(hyperparams))
    if k == len(xdata):
        crossvalType = 'leave-one-out'
    else:
        crossvalType = '{0}-fold'.format(k)
    print 'Running {type} cross-validation to determine good hyperparameters'\
                .format(type=crossvalType)
    print '  Epochs:', epochs

    # Perform the cross validation
    xpieces = []
    ypieces = []
    # This effectively rounds up rather than down
    pieceSize = (len(xdata) + k - 1) / k
    for i in range(k):
        xpieces.append(xdata[i*pieceSize : (i+1)*pieceSize])
        ypieces.append(ydata[i*pieceSize : (i+1)*pieceSize])
    params_k_product = list(itertools.product(hyperparams, range(k)))
    percentsDict, trainPercentsDict = trainfunc(
        cls, xpieces, ypieces, epochs, batch_size, params_k_product
        )
    test_percents = [percentsDict[x] for x in hyperparams]
    train_percents = [trainPercentsDict[x] for x in hyperparams]

    # Print the results
    crossValPrintableTable = [names + [
        'Testing Percent',
        'Training Percent',
        ]]
    columns = zip(*hyperparams) + [
        test_percents,
        train_percents,
        ]
    crossValPrintableTable.extend(zip(*columns))
    print
    printTable(sys.stdout, crossValPrintableTable,
               'Average over all cross-validation runs')
    print

    maxIndex = 0
    for i in xrange(len(test_percents)):
        if test_percents[i] > test_percents[maxIndex]:
            maxIndex = i
    return hyperparams[maxIndex]

def _train(cls, xpieces, ypieces, epochs, batch_size, params_k_product):
    '''
    Performs training sequentially on the main thread

    @return two dictionaries, (test_percents, train_percents)
        (param -> percent) where percent is between 0 and 1
    '''
    percentsDict = {}
    trainPercentsDict = {}
    k = len(xpieces)
    dim = len(xpieces[0][0])
    for params, idx in params_k_product:
        print '  {0}-{1}: '.format(params, idx),
        sys.stdout.flush()
        if params not in percentsDict:
            percentsDict[params] = 0
            trainPercentsDict[params] = 0
        classifier = cls(dim, *params)
        accuracy, train_accuracy = _train_oneiter(
            classifier, xpieces, ypieces,
            epochs, batch_size, idx
            )
        percentsDict[params] += accuracy
        trainPercentsDict[params] += train_accuracy
        print ' ', accuracy, '  ', train_accuracy

    for params in percentsDict:
        percentsDict[params] /= k
        trainPercentsDict[params] /= k

    return (percentsDict, trainPercentsDict)

def crossvalidate(cls, xdata, ydata, k, epochs, batch_size, hyperparams, names):
    '''
    Runs cross-validation for the experiment.  Prints the results, and picks
    and returns the hyper-parameters that have the best percentage (as a
    tuple).

    @params hyperparams - list of tuples, each tuple containing a set of
        hyper-parameters in the order than cls needs them in
    @params names - names of the hyper-parameters for printing purposes
    @return hyper-param tuple containing the best set after cross-validation
    '''
    return _crossvalidate_internal(cls, xdata, ydata, k, epochs, batch_size,
                                   hyperparams, names, _train)

