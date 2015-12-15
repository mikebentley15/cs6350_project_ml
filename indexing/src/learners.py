'''
Implementation of some learning algorithms
'''

import TrainingExample
from crossvalidate import crossvalidate

import numpy as np
import theano
import theano.tensor as T

import argparse
import itertools
import os
import sys

class Sgd(object):
    '''
    Base class for all stochastic gradient descent algorithms
    '''

    def __init__(self, cost_gen, dim_in, dim_out, r, x=None):
        '''
        @param cost_gen
                    A function for generating the cost function.  It needs to
                    be a function of x, y, w, and b. (python function)
                    Example:
                    def cost(x, y, w, b):
                        "Returns a logistic loss cost function"
                        prob = T.nnet.softmax(T.dot(x, w) + b)
                        return -T.mean(T.log(prob)[T.arange(y.shape[0]), y])
        @param dim_in
                    Dimension size of the input
        @param dim_out
                    Dimension size of the output
        @param r    Learning rate (float)
        @param x    Input variable.  If None, then a new one is created
                    (i.e. it would be the first in the pipeline)
        '''
        self.r = r

        self.x = T.matrix('x') if x is None else x
        self.y = T.ivector('y')

        self.w = theano.shared(
            value=np.zeros(
                (dim_in, dim_out),
                dtype=theano.config.floatX
            ),
            name='w',
            borrow=True
            )

        self.b = theano.shared(
            value=np.zeros(
                (dim_out),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
            )

        self.cost = cost_gen(self.x, self.y, self.w, self.b)
        self.params = [self.w, self.b]
        dcost_dw = T.grad(cost=self.cost, wrt=self.w)
        dcost_db = T.grad(cost=self.cost, wrt=self.b)
        self.updates = [
            (self.w, self.w - self.r * dcost_dw),
            (self.b, self.b - self.r * dcost_db),
            ]
        self.output = T.sgn(self.x.dot(self.w) + self.b)

    def train(self, xdata, ydata, epochs, batchSize):
        '''
        @param xdata Data to use for x (2D list or np.array)
        @param ydata Data to use for y (1D list or np.array)
        @param epochs Number of epochs to perform
        @param batchSize Number of samples to send in each iteration of SGD
        '''
        xdata = np.asarray(xdata, dtype=theano.config.floatX) # Make copies
        ydata = np.asarray(ydata, dtype=np.int32)
        xlen = xdata.shape[0]
        # This effectively rounds up instead of down
        batchCount = (xlen + batchSize - 1) / batchSize

        xdata_share = theano.shared(xdata, borrow=True)
        ydata_share = theano.shared(ydata, borrow=True)

        index = T.lscalar()
        trainingFunction = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: xdata_share[index*batchSize : (index+1)*batchSize],
                self.y: ydata_share[index*batchSize : (index+1)*batchSize],
                }
            )

        for epoch in xrange(epochs):
            # Permute the data arrays
            perm = np.random.permutation(xlen)
            xdata[:] = xdata[perm]
            ydata[:] = ydata[perm]
            for minibatchIndex in xrange(batchCount):
                # This is where training actually occurs
                trainingFunction(minibatchIndex)
            # Print a period at every 10% done
            if epoch % max(1, (epochs / 10)) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

def perceptron_loss(x, y, w, b):
    '''
    Returns a perceptron loss function

    @param x  Features (theano matrix)
    @param y  Label (theano ivector)
    @param w  Weight matrix (theano matrix)
    @param b  Bias vector (theano vector)
    @return The perceptron loss function with x, y, w, and b inside
    '''
    return T.sum(T.maximum(0, - y * (x.dot(w) + b).transpose()))

class Perceptron(Sgd):
    '''
    Implements the vanilla perceptron algorithm.

    Note: a bias term is already present.
    '''

    def __init__(self, dim, r):
        '''
        @param dim Number of dimensions in the input
        @param r   Learning rate
        '''
        super(self.__class__, self).__init__(perceptron_loss, dim, 1, r)

    def predict(self, xdata):
        '''
        Takes an itterable containing the data as a 2D array.

        Returns a list of labels of -1 or +1.  There is a small probability of
        getting a label of 0 which happens if the point is exactly on the
        hyperplane.

        @param xdata Feature list to classify
        @return list of labels
        '''
        answers = T.sgn(self.output).eval({self.x : xdata})
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((xdata.shape[0],))

class AveragedPerceptron(Sgd):
    '''
    Implements the averaged perceptron algorithm.

    Note: a bias term is already present.
    '''

    def __init__(self, dim, r):
        '''
        @param dim Number of dimensions in the input
        @param r   Learning rate
        '''
        super(self.__class__, self).__init__(perceptron_loss, dim, 1, r)

        self.w_avg = theano.shared(
            value=np.zeros(
                (dim, 1),
                dtype=theano.config.floatX
            ),
            name='w_avg',
            borrow=True
            )

        self.b_avg = theano.shared(
            value=np.zeros(
                (1),
                dtype=theano.config.floatX
            ),
            name='b_avg',
            borrow=True
            )

        self.updates.extend((
            (self.w_avg, self.w_avg + self.w * self.x.shape[0]),
            (self.b_avg, self.b_avg + self.b * self.x.shape[0]),
            ))


    def train(self, xdata, ydata, epochs, batchSize):
        '''
        Calls the base class train() method and then does post-processing
        '''
        super(self.__class__, self).train(xdata, ydata, epochs, batchSize)
        self.w_avg = self.w_avg / len(xdata)
        self.b_avg = self.b_avg / len(xdata)

    def predict(self, xdata):
        '''
        Takes an itterable containing the data as a 2D array.

        Returns a list of labels of -1 or +1.  There is a small probability of
        getting a label of 0 which happens if the point is exactly on the
        hyperplane.

        @param xdata Feature list to classify
        @return list of labels
        '''
        answers = T.sgn(self.output).eval({self.x : xdata})
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((xdata.shape[0],))

def l1_norm(*arrays):
    'Returns the L1 norm of a list of passed in arrays'
    return sum(abs(x).sum() for x in arrays)

def l2_norm(*arrays):
    'Returns the L2 norm of a list of passed in arrays'
    return sum((x**2).sum() for x in arrays)

def svm_cost(x, y, w, b, C):
    'Returns the cost function of the SVM'
    return C*l2_norm(w) + svm_loss(x, y, w, b)

def svm_loss(x, y, w, b):
    '''
    Returns a perceptron loss function

    @param x  Features (theano matrix)
    @param y  Label (theano ivector)
    @param w  Weight matrix (theano matrix)
    @param b  Bias vector (theano vector)
    @return The perceptron loss function with x, y, w, and b inside
    '''
    return T.maximum(0, 1 - y * (x.dot(w) + b).transpose()).sum()

class SVM(Sgd):
    '''
    Implements the SVM algorithm.

    Note: a bias term is already present.
    '''

    def __init__(self, dim, C, r):
        '''
        @param dim Number of dimensions in the input
        @param r   Learning rate
        @param C   First Step
        '''
        self.C = C
        my_loss = lambda x, y, w, b: svm_cost(x, y, w, b, self.C)
        self.r0 = r
        r = theano.shared(r, name='r')
        self.t = theano.shared(0,  name='t')
        super(self.__class__, self).__init__(my_loss, dim, 1, r)
        self.updates.append((self.r, self.r0 / (1 + self.r0 * self.t * C)))
        self.updates.append((self.t, self.t + 1))

    def predict(self, xdata):
        '''
        Takes an itterable containing the data as a 2D array.

        Returns a list of labels of -1 or +1.  There is a small probability of
        getting a label of 0 which happens if the point is exactly on the
        hyperplane.

        @param xdata Feature list to classify
        @return list of labels
        '''
        xdata_share = theano.shared(
            np.asarray(xdata, dtype=theano.config.floatX),
            borrow=True
            )
        answers = T.sgn(xdata_share.dot(self.w) + self.b).eval()
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((xdata_share.get_value(borrow=True).shape[0],))

class MlpHiddenLayer(object):
    def __init__(self, dim_in, dim_out, x=None, activation=T.tanh):
        '''
        The hidden layer used in Multi-Layered Perceptron (MLP).  The two
        layers are fully connected.  The two different supported activation
        functions are T.tanh and T.nnet.sigmoid.  You could pass in a different
        activation function, but the initialization will match what would have
        been done for tanh.

        @param dim_in: dimensionality of input
        @param dim_out: dimensionality of output
        @param r: hyper-parameter learning-rate
        @param x: (theano.tensor.dmatrix) input matrix, or None if this is the
            beginning layer
        @param activation: (theano.Op or function) Non linearity to be applied
            in the hidden layer
        '''
        w_range = np.sqrt(6. / (dim_in + dim_out))
        self.w = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    low=-w_range,
                    high=w_range,
                    size=(dim_in, dim_out),
                    ),
                dtype=theano.config.floatX,
                ),
            name='w',
            borrow=True,
            )
        self.b = theano.shared(
            value=np.zeros(
                (dim_out,),
                dtype=theano.config.floatX,
                ),
            name='b',
            borrow=True,
            )
        if x is None:
            x = T.matrix('x')
        self.x = x
        self.output = activation(x.dot(self.w) + self.b)
        self.params = [self.w, self.b]


def mlp_cost(x, y, w, b, C1, C2, w_hidden, b_hidden):
    '''
    @param x: input features
    @param y: correct labels of -1 or 1
    @param w: weight vector
    @param b: bias term
    @param C1: constant in front of the L1 loss
    @param C2: constant in front of the L2 loss
    @param w_hidden: hidden layer weight matrix
    @param b_hidden: hidden layer bias vector
    '''
    # TODO: replace svm_loss with some other loss
    return (
        C1 * l1_norm(w, w_hidden)
        + C2 * l2_norm(w, w_hidden)
        + svm_loss(x, y, w, b)
        )


class Mlp(Sgd):
    def __init__(self, dim_in, dim_hidden, dim_out, r, C1, C2, x=None):
        '''
        @param dim_in: number of input features
        @param dim_hidden: number of nodes in the hidden layer
        @param dim_out: number of outputs
        @param r: learning rate
        @param x: (theano.tensor.TensorType - one minibatch) symbolic variable
            for the input features.  If none, then it is assumed that this is
            the first layer and a variable will be created.
        '''
        self.hiddenLayer = MlpHiddenLayer(
            dim_in,
            dim_hidden,
            x=x,
            activation=T.tanh,
            )
        self.C1 = C1
        self.C2 = C2
        self.r0 = r
        my_cost = lambda xin, y, w, b: (
            mlp_cost(xin, y, w, b, self.C1, self.C2,
                     self.hiddenLayer.w, self.hiddenLayer.b)
            )
        self.t = theano.shared(0, name='t')
        r = theano.shared(r, name='r')
        super(self.__class__, self).__init__(
            my_cost,
            dim_hidden,
            dim_out,
            r,
            x=self.hiddenLayer.output
            )
        # Use a different input than what it set in the base class
        self.x = self.hiddenLayer.x
        self.params.extend(self.hiddenLayer.params)
        dcost_dhw = T.grad(cost=self.cost, wrt=self.hiddenLayer.w)
        dcost_dhb = T.grad(cost=self.cost, wrt=self.hiddenLayer.b)
        self.updates.extend([
            (self.hiddenLayer.w, self.hiddenLayer.w - self.r * dcost_dhw),
            (self.hiddenLayer.b, self.hiddenLayer.b - self.r * dcost_dhb),
            (self.r, self.r0 / (1 + self.r0 * self.t * self.C2)),
            (self.t, self.t + 1),
            ])

    def predict(self, xdata):
        '''
        Takes an itterable containing the data as a 2D array.

        Returns a list of labels of -1 or +1.  There is a small probability of
        getting a label of 0 which happens if the point is exactly on the
        hyperplane.

        @param xdata Feature list to classify
        @return list of labels
        '''
        answers = T.sgn(self.hiddenLayer.output.dot(self.w) + self.b).eval({self.x: xdata})
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((len(xdata),))

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Theano implementation of the vanilla Perceptron, and maybe other
        traininers too.
        ''')
    parser.add_argument('-r', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=5)
    parser.add_argument(
        '-d', '--base-dir',
        default='/home/pontsler/Documents',
        help='''
        Directory where to find the data from the ML class (data0 and data1).
        ''')
    return parser.parse_args(args=arguments)

def main(arguments):
    'Main entry point'
    args = parseArgs(arguments)

    r = args.learning_rate
    epochs = args.epochs
    batchSize = args.batch_size
    print 'learning rate fixed:      r =', r
    print 'epochs fixed:        epochs =', epochs
    print 'batch size fixed:     batch =', batchSize

    basedir = '/home/pontsler/Documents'

    sets = [
        # Name         training path       testing path
        #('sanity   ', 'sanityCheck-train.dat', 'sanityCheck-train.dat'),
        #('train0.10', 'data0/train0.10', 'data0/test0.10'),
        #('train0.20', 'data0/train0.20', 'data0/test0.20'),
        ('train1.10', 'data1/train1.10', 'data1/test1.10'),
        ]

    for name, trainingPath, testingPath in sets:
        training = TrainingExample.fromSvm(
            os.path.join(args.base_dir, trainingPath)
            )
        testing = TrainingExample.fromSvm(
            os.path.join(args.base_dir, testingPath)
            )
        testPerceptron(name, training, testing,
                       crossepochs=10,
                       epochs=epochs,
                       batchSize=batchSize)
    print

def testPerceptron(name, trainExamples, testExamples, crossepochs=10,
                   epochs=10, batchSize=1):
    '''
    Trains an averaged Perceptron classifier from the training examples and
    then calculates the accuracy of the generated classifier on the test
    examples.

    Prints out the results to the console
    '''
    rvalues = [0.01, 0.05, 0.1, 0.5]
    Cvalues = [0.001, 0.005, 0.01, 0.05]
    dimvalues = [10, 20]

    #learner = Perceptron
    #learner = AveragedPerceptron
    #hypers = [(x,) for x in rvalues]
    #names = ['r']
    #learner = SVM
    #hypers = list(itertools.product(rvalues, Cvalues))
    #names = ['r', 'C']
    learner = lambda dim_in, dim_hidden, r, C2: Mlp(dim_in, dim_hidden, 1, r, 0, C2)
    hypers = list(itertools.product(dimvalues, rvalues, Cvalues))
    names = ['hidden-dimension', 'r', 'C']

    featuresList = [x.features for x in trainExamples]
    labels = [x.label for x in trainExamples]
    testFeatures = [x.features for x in testExamples]
    testLabels = [x.label for x in testExamples]
    k = 5
    print 'Performing cross-validation'
    print '  k:              ', k
    print '  parameters:     ', names
    bestHyper = crossvalidate(learner, featuresList, labels, k, crossepochs,
                              batchSize, hypers, names)
    print '  best params:    ', names, '=', bestHyper
    print

    p = learner(len(featuresList[0]), *bestHyper)
    print 'training ' + name + ' ',
    sys.stdout.flush()
    p.train(featuresList, labels, epochs, batchSize)
    print ' done'

    # Test accuracy on the training set
    predictions = p.predict(featuresList)
    accuracy = np.sum(labels == predictions) / float(len(labels))
    print name, '  train accuracy:  ', accuracy

    # Test accuracy on the testing set
    predictions = p.predict(testFeatures)
    accuracy = np.sum(testLabels == predictions) / float(len(testLabels))
    print name, '  test accuracy:   ', accuracy

if __name__ == '__main__':
    main(sys.argv[1:])

