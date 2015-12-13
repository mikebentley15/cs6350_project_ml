'''
Script for training on the LDS data
'''

import imagefuncs
from learners import Perceptron, AveragedPerceptron, SVM
from printTable import printTable
from outdup import OutDuplicator

from collections import namedtuple
from fractions import Fraction
import argparse
import csv
import itertools
import os
import random
import resource
import sys
import time

import numpy as np

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Trains a classifier on the provided training data and prints testing results.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-cache', required=True, help='Dir containing training image cache')
    parser.add_argument('--test-cache', required=True, help='Dir containing testing image cache')
    parser.add_argument('--train', required=True, help='path for tsv containing training labels')
    parser.add_argument('--test', required=True, help='path for tsv containing training labels')
    parser.add_argument('-o', '--output', default='classifier.dat', help='output file for the classifier')
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=5)
    return parser.parse_args(args=arguments)

def main(arguments):
    'Main entry point'
    args = parseArgs(arguments)
    #print args

    print 'Loading data ...      ',
    sys.stdout.flush()
    start = time.clock()
    xdata, ydata = loadrecords(args.train, args.train_cache)
    xdata = np.reshape(xdata, (xdata.shape[0], xdata.shape[1] * xdata.shape[2]))
    print 'done'
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  elapsed time:       ', time.clock() - start
    print '  memory used:        ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print '  xshape:             ', xdata.shape
    print '  yshape:             ', ydata.shape
    print

    print 'Perform cross-validation'
    k = 15
    rValues = [0.0001, 0.001, 0.01, 0.1, 0.5]
    #CValues = [0.0001, 0.001, 0.01, 0.1, 1, 10, 40, 100]
    print 'Possible r values:    ', rValues
    #print 'Possible C values:    ', CValues
    print 'Doing {k}-cross validation'.format(k=k)
    #hyperparams = list(itertools.product(rValues, CValues))
    #hypernames = ('r', 'C')
    hyperparams = [(x,) for x in rValues]
    hypernames = ('r',)
    start = time.clock()
    #r, C = crossvalidate(SVM, xdata, ydata, k, args.epochs, args.batch_size,
    #                     hyperparams, hypernames)
    r = crossvalidate(AveragedPerceptron, xdata, ydata, k, args.epochs, args.batch_size,
                      hyperparams, hypernames)
    print '  elapsed time:       ', time.clock() - start
    print '  best r:             ', r
    #print '  best C:             ', C
    print

    print 'Training Perceptron'
    print '  epochs:             ', args.epochs
    print '  batch size:         ', args.batch_size
    print '  training  ',
    sys.stdout.flush()
    start = time.clock()
    #p = Perceptron(xdata.shape[1], args.learning_rate)
    p = AveragedPerceptron(xdata.shape[1], r)
    #p = SVM(xdata.shape[1], C, r)
    p.train(xdata, ydata, args.epochs, args.batch_size)
    print ' done'
    predictions = p.predict(xdata)
    print '  elapsed time:       ', time.clock() - start
    print '  training accuracy:  ', np.sum(ydata == predictions) / float(len(ydata))
    del xdata
    del ydata
    print

    print 'Loading test data ... ',
    sys.stdout.flush()
    start = time.clock()
    testx, testy = loadrecords(args.test, args.test_cache)
    testx = np.reshape(testx, (testx.shape[0], testx.shape[1] * testx.shape[2]))
    print 'done'
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  elapsed time:       ', time.clock() - start
    print '  memory used:        ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print '  xshape:             ', testx.shape
    print '  yshape:             ', testy.shape
    print

    print 'Running classifier on test set'
    start = time.clock()
    predictions = p.predict(testx)
    print '  elapsed time:       ', time.clock() - start
    print '  test accuracy:      ', np.sum(testy == predictions) / float(len(testy))
    print

def crossvalidate(cls, xdata, ydata, k, epochs, batch_size, hyperparams, names):
    '''
    Runs cross-validation for the experiment.  Prints the results, and picks
    and returns the hyper-parameters that have the best percentage (as a
    tuple).
    '''
    percents = np.zeros(len(hyperparams))
    if k == len(xdata):
        type = 'leave-one-out'
    else:
        type = '{}-fold'.format(k)
    print 'Running {type} cross-validation to determine good hyperparameters'\
                .format(type=type)
    print '  Epochs:', epochs

    # Perform the cross validation
    random.shuffle(xdata)  # Shuffle before splitting, but use the same shuffle for each hyper-param
    for params in hyperparams:
        testRunResults = []
        print ' ', params,
        sys.stdout.flush()
        for crossvalIdx in range(k):
            classifier = cls(xdata.shape[1], *params)
            testidx = np.arange(crossvalIdx, len(ydata), k)
            testx = xdata[testidx]
            testy = ydata[testidx]
            trainidx = [x for x in xrange(len(ydata)) if x not in testidx]
            trainx = xdata[trainidx]
            trainy = ydata[trainidx]
            for _ in range(epochs):
                perm = np.random.permutation(len(trainy))
                trainx[:] = trainx[perm]
                trainy[:] = trainy[perm]
                origstdout = sys.stdout
                try:
                    with open(os.devnull, 'w') as devnull:
                        sys.stdout = devnull
                        classifier.train(trainx, trainy, epochs, batch_size)
                finally:
                    sys.stdout = origstdout
                sys.stdout.write('.')
                sys.stdout.flush()
            predictions = classifier.predict(testx)
            accuracy = np.sum(testy == predictions) / float(len(testy))
            testRunResults.append(accuracy)
        percents.append(sum(x for x in testRunResults)/len(testRunResults))
        print ' ', percents[-1]
    print

    # Print the results
    crossValPrintableTable = [names + [
        'Average Percent',
        ]]
    columns = zip(*hyperparams) + [
        percents,
        ]
    crossValPrintableTable.extend(zip(*columns))
    print
    printTable(sys.stdout, crossValPrintableTable, 'Average over all cross-validation runs')
    print

    maxIndex = 0
    for i in xrange(len(percents)):
        if percents[i] > percents[maxIndex]:
            maxIndex = i
    return hyperparams[maxIndex]


def loadrecords(csvfile, cachedir):
    '''
    Loads the records described by the csv file and cropped images in the
    cachedir.
    This returns xdata as a numpy 3D array, which is an array of 2D images.
    The ydata will be a numpy 1D array of values of 1 or -1.  The ydata array
    represents the sex field of the csv file.  +1 represents male and -1
    represents female.
    >>> xdata, ydata = loadrecords(csvfile, cachedir)
    '''
    xdata = []
    ydata = []
    with open(csvfile, 'r') as trainFile:
        reader = csv.DictReader(trainFile, dialect='excel-tab')
        Record = namedtuple('Record', 'imagepath line sex')
        for line in reader:
            filename = os.path.basename(line['imagePath'])
            split = os.path.splitext(filename)
            cachepic = split[0] + '-' + line['line'] + '.png'
            xdata.append(imagefuncs.loadImage(os.path.join(cachedir, cachepic)))
            ydata.append(1 if line['truth-sex'] == 'M' else -1)
    return np.asarray(xdata), np.asarray(ydata)

if __name__ == '__main__':
    origstdout = sys.stdout
    try:
        with open('run.log', 'w') as runlog:
            sys.stdout = OutDuplicator([runlog, origstdout])
            main(sys.argv[1:])
    finally:
        sys.stdout = origstdout
