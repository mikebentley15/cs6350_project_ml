'''
Script for training on the LDS data
'''

#import imagefuncs
from learners import Perceptron, AveragedPerceptron, SVM
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

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Trains a classifier on the provided training data and prints testing results.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train', help='path for pickle file containing training cache')
    parser.add_argument('test', help='path for pickle containing test data cache')
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
    with open(args.train, 'r') as trainfile:
        xdata, ydata = pickle.load(trainfile)
    xdata = np.reshape(xdata, (xdata.shape[0], xdata.shape[1] * xdata.shape[2]))
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
    #CValues = [0.0001, 0.001, 0.01, 0.1, 1, 10, 40, 100]
    print 'Possible r values:    ', rValues
    #print 'Possible C values:    ', CValues
    #hyperparams = list(itertools.product(rValues, CValues))
    #hypernames = ('r', 'C')
    hyperparams = [(x,) for x in rValues]
    hypernames = ['r',]
    #start = time.clock()
    #r, C = crossvalidate(SVM, xdata, ydata, k, args.epochs, args.batch_size,
    #                     hyperparams, hypernames)
    start = time.clock()
    print 'Doing {k}-cross validation sequentially'.format(k=k)
    r, = crossvalidate(AveragedPerceptron, xdata, ydata, k, args.epochs, args.batch_size,
                       hyperparams, hypernames)
    print '  elapsed time:       ', time.clock() - start
    print '  best r:             ', r
    # I tried to get threading to work, but it just didn't do very well.
    # It took a lot longer than single threaded, maybe because theano can't be
    # parallelized like that...
    #print 'Doing {k}-cross validation multithreaded'.format(k=k)
    #r, = crossvalidate_threaded(AveragedPerceptron, xdata, ydata, k,
    #                            args.epochs, args.batch_size,
    #                            hyperparams, hypernames)
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used before gc:', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    gc.collect()
    kb_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print '  memory used after gc: ', kb_used, 'KB,', float(kb_used) / 2**10, 'MB'
    print

    print 'Training Perceptron'
    print '  epochs:             ', args.epochs
    print '  batch size:         ', args.batch_size
    print '  training  ',
    sys.stdout.flush()
    start = time.clock()
    #p = Perceptron(xdata.shape[1], r)
    p = AveragedPerceptron(xdata.shape[1], r)
    #p = SVM(xdata.shape[1], C, r)
    p.train(xdata, ydata, args.epochs, args.batch_size)
    print ' done'
    predictions = p.predict(xdata)
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
    with open(args.test, 'r') as testfile:
        testx, testy = pickle.load(testfile)
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

if __name__ == '__main__':
    origstdout = sys.stdout
    try:
        with open('run.log', 'w') as runlog:
            sys.stdout = OutDuplicator([runlog, origstdout])
            main(sys.argv[1:])
    finally:
        sys.stdout = origstdout
