'''
Script for training on the LDS data
'''

#import imagefuncs
from learners import Perceptron, AveragedPerceptron, SVM, Mlp, LogisticRegression, ConvNet
from outdup import OutDuplicator
from crossvalidate import crossvalidate

import argparse
import gc
import itertools
import pickle
import resource
import sys
import time

import numpy as np
import theano

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Trains a classifier on the provided training data and prints testing results.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train', help='path for pickle file containing training cache')
    parser.add_argument('test', help='path for pickle containing test data cache')
    parser.add_argument('-o', '--output', default='classifier.pkl', help='output file for the classifier')
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=5)
    parser.add_argument('-l', '--log', default='run.log', help='Where to duplicate stdout')
    parser.add_argument('-c', '--classifier', default='SVM', help='''
        Which classifier to train.  The choices are "SVM", "Perceptron",
        "AveragedPerceptron", "LogisticRegression", "MLP", and "ConvNet".
        ''')
    parser.add_argument('--crossval-epochs', type=int, default=10)
    return parser.parse_args(args=arguments)

def main(arguments):
    'Main entry point'
    args = parseArgs(arguments)
    origstdout = sys.stdout
    try:
        with open(args.log, 'w') as runlog:
            sys.stdout = OutDuplicator([runlog, origstdout])
            classifier = _runExperiment(args.train, args.test, args.crossval_epochs,
                                        args.epochs, args.batch_size, args.classifier)
            with open(args.output, 'w') as classifierfile:
                pickle.dump(classifier, classifierfile)
    finally:
        sys.stdout = origstdout

def _preprocess(xdata, reshape=True):
    '''
    Performs necessary preprocessing to improve learning and to shape the data
    how it needs to be.
    '''
    floatX = eval('np.' + theano.config.floatX)
    xdata = floatX(xdata) / xdata.max() # Scale it between 0 and 1
    xdata = 1 - xdata # Invert it to have majority small values
    #if reshape:
    #    xdata = np.reshape(xdata, (xdata.shape[0], xdata.shape[1] * xdata.shape[2]))
    xdata = np.reshape(xdata, (xdata.shape[0], xdata.shape[1] * xdata.shape[2]))
    return xdata

def _runExperiment(train, test, cross_epochs, epochs, batch_size, classifierName):
    '''
    Runs the experiment returning the classifier generated
    '''
    print 'Running the experiment'
    print '  training data:      ', train
    print '  testing data:       ', test
    print '  epochs:             ', epochs
    print '  batch size:         ', batch_size
    print '  classifier:         ', classifierName
    print

    print 'Loading data ...      ',
    sys.stdout.flush()
    start = time.clock()
    with open(train, 'r') as trainfile:
        xdata, ydata = pickle.load(trainfile)
    xdata = _preprocess(xdata, reshape=(classifierName != 'ConvNet'))
    # Split into verify set and data set
    xlen = len(xdata)
    xverify = xdata[:xlen/5]
    yverify = ydata[:xlen/5]
    xdata = xdata[xlen/5:]
    ydata = ydata[xlen/5:]
    print 'done'
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  elapsed time:       ', time.clock() - start
    print '  memory used:        ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print '  xshape:             ', xdata.shape
    print '  yshape:             ', ydata.shape
    print

    print 'Perform cross-validation'
    k = 5
    rValues = [0.0001, 0.001, 0.01, 0.1, 0.5]
    CValues = [0.0001, 0.001, 0.01, 0.1]
    hiddenDims = [1, 20, 40, 80]
    out_im1_amounts = [10, 20]
    out_im2_amounts = [20, 50]

    rValues = [0.05]
    CValues = [0.01]
    hiddenDims = [20]
    out_im1_amounts = [10]
    out_im2_amounts = [20]

    mlpLearner = lambda dim_in, dim_hidden, r, C: \
        Mlp(dim_in, dim_hidden, 2, r, C)
    im_shape = (40, 68)
    convnetLearner = lambda dim_in, out_im1, out_im2, dim_hidden, r, C: \
        ConvNet(
            im_shape,
            (out_im1, out_im2),
            ((5, 5), (5, 5)),
            ((2, 2), (2, 2)),
            batch_size,
            dim_hidden,
            2,
            r,
            C,
            )
    classifierMap = {
        'MLP': (
            mlpLearner,
            list(itertools.product(hiddenDims, rValues, CValues)),
            ['hidden-dims', 'r', 'C']
            ),
        'SVM': (SVM, list(itertools.product(rValues, CValues)), ['r', 'C']),
        'Perceptron': (Perceptron, [(x,) for x in rValues], ['r']),
        'AveragedPerceptron': (
            AveragedPerceptron,
            [(x,) for x in rValues],
            ['r']
            ),
        'LogisticRegression': (
            LogisticRegression,
            list(itertools.product([2], rValues, CValues)),
            ['dim_out', 'r', 'C']
            ),
        'ConvNet': (
            convnetLearner,
            list(itertools.product(out_im1_amounts, out_im2_amounts, hiddenDims, rValues, CValues)),
            ['kernel_1', 'kernel_2', 'hidden-dims', 'r', 'C']
            ),
        }
    algorithm, hyperparams, hypernames = classifierMap[classifierName]
    if classifierName in ('LogisticRegression', 'MLP'):
        ydata -= ydata.min()
        ydata /= ydata.max()
        yverify -= yverify.min()
        yverify /= yverify.max()
    start = time.clock()
    print 'Doing {k}-cross validation sequentially'.format(k=k)
    print '  params:             ', hypernames
    for i in xrange(len(hypernames)):
        print '  {0} values:    '.format(hypernames[i]), sorted(set([x[i] for x in hyperparams]))
    params = crossvalidate(algorithm, xdata, ydata, k, cross_epochs, batch_size,
                           hyperparams, hypernames)
    print '  elapsed time:       ', time.clock() - start
    print '  best params:        ', params
    # I tried to get threading to work, but it just didn't do very well.
    # It took a lot longer than single threaded, maybe because theano can't be
    # parallelized like that...
    #print 'Doing {k}-cross validation multithreaded'.format(k=k)
    #params = crossvalidate_threaded(AveragedPerceptron, xdata, ydata, k,
    #                            epochs, batch_size,
    #                            hyperparams, hypernames)
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used before gc:', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    gc.collect()
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used after gc: ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print

    print 'Training Perceptron'
    print '  epochs:             ', epochs
    print '  batch size:         ', batch_size
    print '  training  ',
    sys.stdout.flush()
    start = time.clock()
    classifier = algorithm(xdata.shape[1], *params)
    classifier.train(xdata, ydata, epochs, batch_size, xverify=xverify, yverify=yverify)
    print ' done'
    predictions = classifier.predict(xdata)
    print '  elapsed time:       ', time.clock() - start
    print '  training accuracy:  ', np.sum(ydata == predictions) / float(len(ydata))
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used before gc:', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    del xdata
    del ydata
    gc.collect()
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used after gc: ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print

    print 'Loading test data ... ',
    sys.stdout.flush()
    start = time.clock()
    with open(test, 'r') as testfile:
        testx, testy = pickle.load(testfile)
    testx = _preprocess(testx, reshape=(classifierName != 'ConvNet'))
    if classifierName in ('LogisticRegression', 'MLP'):
        testy -= testy.min()
        testy /= testy.max()
    print ' done'
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  elapsed time:       ', time.clock() - start
    print '  memory used:        ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print '  xshape:             ', testx.shape
    print '  yshape:             ', testy.shape
    print

    print 'Running classifier on test set'
    start = time.clock()
    predictions = classifier.predict(testx)
    print '  elapsed time:       ', time.clock() - start
    print '  test accuracy:      ', np.sum(testy == predictions) / float(len(testy))
    print

    return classifier

if __name__ == '__main__':
    main(sys.argv[1:])
