'''
Implementation of some learning algorithms
'''

import TrainingExample

import numpy as np
import theano
import theano.tensor as T

import argparse
import os
import random
import sys

class Sgd(object):
    def __init__(self, cost_gen, dim_in, dim_out, r, x=None):
        '''
        @param cost_gen
                    A function for generating the cost function.  It needs to
                    be a function of x, y, w, and b. (python function)
                    Example:
                    def cost(x, y, w, b):
                        'Returns a logistic loss cost function'
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
        self.dcost_dw = T.grad(cost=self.cost, wrt=self.w)
        self.dcost_db = T.grad(cost=self.cost, wrt=self.b)
        self.updates = [
            (self.w, self.w - self.r * self.dcost_dw),
            (self.b, self.b - self.r * self.dcost_db),
            ]

    def train(self, xdata, ydata, epochs, batchSize):
        '''
        @param xdata Data to use for x (2D list or np.array)
        @param ydata Data to use for y (1D list or np.array)
        @param epochs Number of epochs to perform
        @param batchSize Number of samples to send in each iteration of SGD
        '''
        xdata = np.asarray(xdata, dtype=theano.config.floatX)
        ydata = np.asarray(ydata, dtype=np.int32)
        xlen = xdata.shape[0]
        batchCount = xlen / batchSize
        assert xlen % batchSize == 0, 'Example set is not divisible by batchSize'

        xdata_share = theano.shared(xdata, borrow=True)
        ydata_share = theano.shared(ydata, borrow=True)


        # TODO: Try to figure out how to deal with batchSize values that don't
        # TODO:   need to exactly divide input count.
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
            #print 'epoch', epoch+1
            #print '  w', self.w.get_value().reshape(len(xdata[0]))
            #print '  b', self.b.get_value()
            #print '  xdata', xdata
            #print '  ydata', ydata
            for minibatchIndex in xrange(batchCount):
                #print '  minibatch', minibatchIndex+1
                #print '    xdata[{0}]'.format(minibatchIndex), xdata[minibatchIndex]
                #print '    ydata[{0}]'.format(minibatchIndex), ydata[minibatchIndex]
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
    
    
def SVM_loss(x, y, w, b, C):
    '''
    Returns a perceptron loss function

    @param x  Features (theano matrix)
    @param y  Label (theano ivector)
    @param w  Weight matrix (theano matrix)
    @param b  Bias vector (theano vector)
    @return The perceptron loss function with x, y, w, and b inside
    '''
    return T.sum(T.maximum(0, 1 - y * (x.dot(w) + b).transpose()))    

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
        xdata_share = theano.shared(
            np.asarray(xdata, dtype=theano.config.floatX),
            borrow=True
            )
        answers = T.sgn(xdata_share.dot(self.w) + self.b).eval()
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((xdata_share.get_value(borrow=True).shape[0],))

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
        my_loss = lambda x, y, w, b: SVM_loss(x, y, w, b, self.C)
        super(self.__class__, self).__init__(my_loss, dim, 1, r)
        self.r0 = r
        self.r = theano.shared(
            value=np.array([r],
                dtype=theano.config.floatX
            ),
            name='r',
            borrow=True
            )
        self.t = theano.shared(
            value=np.zeros(1,
                dtype=np.int32
            ),
            name='t',
            borrow=True
            )
        self.updates.append((self.r, self.r0/(1+self.r0*self.t/C)))
        self.updates.append((self.t, self.t+1))

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
        xdata_share = theano.shared(
            np.asarray(xdata, dtype=theano.config.floatX),
            borrow=True
            )
        answers = T.sgn(xdata_share.dot(self.w_avg) + self.b_avg).eval()
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape
        # to a 1D array.
        return answers.reshape((xdata_share.get_value(borrow=True).shape[0],))

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Theano implementation of the vanilla Perceptron, and maybe other
        traininers too.
        ''')
    parser.add_argument('-r', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=5)
    return parser.parse_args(args=arguments)

class HiddenLayer(Sgd):
    def __init__(self, dim_in, dim_out, r, x=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        @param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        @param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        @param n_in: dimensionality of input

        :type n_out: int
        @param n_out: number of hidden units

        :type activation: theano.Op or function
        @param activation: Non linearity to be applied in the hidden
                           layer
        """
        super(self.__class__, self).__init__(perceptron_loss, dim_in, dim_out, r, x=x)
        rand_range = np.sqrt(6. / (dim_in + dim_out))
        w_arr = self.w.get_value(borrow=True)
        w_arr[:] = np.random.uniform(low=-rand_range,
                                     high=rand_range,
                                     size=(dim_in, dim_out))
        if activation == T.nnet.sigmoid:
            w_arr *= 4

        #linear_combination = T.dot(


        #self.input = input
        #L_P=Perceptron(len(featuresList[0]), r)
        #W_values = numpy.asarray(  # Not going to work because w is a shared
        #            rng.uniform(
        #                low=-numpy.sqrt(6. / (n_in + n_out)),
        #                high=numpy.sqrt(6. / (n_in + n_out)),
        #                size=(n_in, n_out)
        #                ),
        #            dtype=theano.config.floatX
        #        )
        #        if activation == theano.tensor.nnet.sigmoid:
        #            L_P.W_values *= 4

        #    W = theano.shared(value=W_values, name='W', borrow=True) ### Not going to work

        #if b is None:
        #    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        #    b = theano.shared(value=b_values, name='b', borrow=True)

        ##self.W = W
        ##self.b = b
        #L_P.w = W
        #L_P.b = b


        #### what to do from here?
        #
        #self.output = (
        #    L_P.out if activation is None
        #    else activation(L_P.out)
        #)
        ## parameters of the model
        #self.params = [L_P.w, L_P.b]
        
                 

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
        training = TrainingExample.fromSvm(os.path.join(basedir, trainingPath))
        testing = TrainingExample.fromSvm(os.path.join(basedir, testingPath))
        testPerceptron(name, training, testing, r, epochs)
    print

def testPerceptron(name, trainExamples, testExamples, r = 0.2, epochs = 10, batchSize = 1):
    '''
    Trains an averaged Perceptron classifier from the training examples and
    then calculates the accuracy of the generated classifier on the test
    examples.

    Prints out the results to the console
    '''
    featuresList = [x.features for x in trainExamples]
    labels = [x.label for x in trainExamples]
    #p = Perceptron(len(featuresList[0]), r)
    #p = AveragedPerceptron(len(featuresList[0]), r)
    p = SVM(len(featuresList[0]),20,.01)

    print 'training ' + name + ' ',
    sys.stdout.flush()
    p.train(featuresList, labels, epochs, batchSize)
    print ' done'
    #print 'w vector:       ', p.w.get_value(borrow=True).reshape(-1).tolist()
    #print 'w_avg vector:   ', p.w_avg.eval().reshape(-1).tolist()
    testFeatures = [x.features for x in testExamples]
    testLabels = [x.label for x in testExamples]

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

