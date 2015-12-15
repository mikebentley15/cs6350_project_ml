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

class _CrossValidationThread(threading.Thread):
    '''
    A thread class used in the multi-threaded cross-validation variants.

    Attributes:
      accuracy     None before finished, a number between 0 and 1 after
      batch_size   Batch size passed to the constructor
      classifier   Classifier created and trained
      epochs       Epochs passed into the constructor
      hyperparams  Hyper-parameters passed into the constructor
      idx          idx passed into the constructor
      xpieces      xpieces passed to the constructor
      ypieces      ypieces passed to the constructor
    '''

    def __init__(self, trainer, hyperparams, xpieces, ypieces, idx, epochs,
                 batch_size):
        '''
        @param trainer      Class for the classifier
            constructor: classifier = trainer(len(xpieces[0][0]), *hyperparams)
            train:       classifier.train(xdata, ydata, epochs, batch_size)
            predict:     predictions = classifier.predict(xtest)
        @param hyperparams  Hyper-parameters to pass into trainer
        @param xpieces      List of broken xdata into k pieces
        @param ypieces      List of broken ydata into k pieces
        @param idx          Index of xpieces for testing of the k-fold
                            cross-validation
        @param epochs       Number of epochs to perform during training
        @param batch_size   Batch size for mini-batch stochastic gradient
                            descent
        '''
        super(self.__class__, self).__init__()
        self.hyperparams = hyperparams
        self.xpieces = xpieces
        self.ypieces = ypieces
        self.classifier = trainer(len(xpieces[0][0]), *self.hyperparams)
        self.idx = idx
        self.epochs = epochs
        self.batch_size = batch_size
        self.accuracy = None
    def run(self):
        '''
        Runs the training assigned to this thread
        '''
        # TODO: get hyperparams and k from shared queue
        self.accuracy = _train_oneiter(
            self.classifier,
            self.xpieces,
            self.ypieces,
            self.epochs,
            self.batch_size,
            self.idx
            )
        print '  {0}-{1}:'.format(self.hyperparams, self.idx), self.accuracy

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

    @return accuracy for this iteration (a number between 0 and 1)
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
    predictions = classifier.predict(xtest)
    accuracy = np.sum(predictions == ytest) / float(len(ytest))
    return accuracy

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
    percents = np.zeros(len(hyperparams))
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
    percentsDict = trainfunc(cls, xpieces, ypieces, epochs, batch_size,
                             params_k_product)
    percents = [percentsDict[x] for x in hyperparams]

    # Print the results
    crossValPrintableTable = [names + [
        'Average Percent',
        ]]
    columns = zip(*hyperparams) + [
        percents,
        ]
    crossValPrintableTable.extend(zip(*columns))
    print
    printTable(sys.stdout, crossValPrintableTable,
               'Average over all cross-validation runs')
    print

    maxIndex = 0
    for i in xrange(len(percents)):
        if percents[i] > percents[maxIndex]:
            maxIndex = i
    return hyperparams[maxIndex]

def _train(cls, xpieces, ypieces, epochs, batch_size, params_k_product):
    '''
    Performs training sequentially on the main thread
    '''
    percentsDict = {}
    k = len(xpieces)
    dim = len(xpieces[0][0])
    for params, idx in params_k_product:
        print '  {0}-{1}: '.format(params, idx),
        sys.stdout.flush()
        if params not in percentsDict:
            percentsDict[params] = 0
        classifier = cls(dim, *params)
        accuracy = _train_oneiter(classifier, xpieces, ypieces,
                                  epochs, batch_size, idx)
        percentsDict[params] += accuracy
        print ' ', accuracy

    for params in percentsDict:
        percentsDict[params] /= k

    return percentsDict

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

def _train_threaded(cls, xpieces, ypieces, epochs, batch_size,
                    params_k_product):
    '''
    Performs training with multiple threads
    '''
    percentsDict = {}
    k = len(xpieces)
    # TODO: define this dynamically, and create a queue instead of a thread per
    #       item
    #threadCount = 8
    threads = [
        _CrossValidationThread(cls, params, xpieces, ypieces, idx, epochs,
                               batch_size)
        for params, idx in params_k_product
        ]
    for thread in threads:
        percentsDict[thread.hyperparams] = 0
        thread.start()
    for thread in threads:
        thread.join()
        percentsDict[thread.hyperparams] += thread.accuracy
    for params in percentsDict:
        percentsDict[params] /= k
    return percentsDict

def crossvalidate_threaded(cls, xdata, ydata, k, epochs, batch_size,
                           hyperparams, names):
    '''
    Runs cross-validation for the experiment.  Prints the results, and picks
    and returns the hyper-parameters that have the best percentage (as a
    tuple).  This variant performs cross-validation using multiple threads.

    @params hyperparams - list of tuples, each tuple containing a set of
        hyper-parameters in the order than cls needs them in
    @params names - names of the hyper-parameters for printing purposes
    @return hyper-param tuple containing the best set after cross-validation
    '''
    return _crossvalidate_internal(cls, xdata, ydata, k, epochs, batch_size,
                                   hyperparams, names, _train_threaded)

