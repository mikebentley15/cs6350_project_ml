'''
Implementation of some learning algorithms
'''

import TrainingExample
from crossvalidate import crossvalidate

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

import argparse
import itertools
import os
import sys

floatX = eval('np.' + theano.config.floatX)

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
                        # Returns a logistic loss cost function
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

    def train(self, xdata, ydata, epochs, batchSize, xverify=None, yverify=None):
        '''
        @param xdata Data to use for x (2D list or np.array)
        @param ydata Data to use for y (1D list or np.array)
        @param epochs Number of epochs to perform
        @param batchSize Number of samples to send in each iteration of SGD
        @param xverify Verify data set used to exit early from training (optional)
        @param yverify Verify data set used to exit early from training (optional)
        '''
        xdata = np.asarray(xdata, dtype=theano.config.floatX).copy() # Make copies
        ydata = np.asarray(ydata, dtype=np.int32).copy()
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
        max_validation_accuracy = 0
        best_vars = None
        early_exit_threshold = 0.05 # Percentage points above minimum validation accuracy
        validation_check = min(5, epochs / 10) # how often to check

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
            #print 'cost: ', self.cost.eval({self.x: xdata, self.y: ydata})
            if xverify is not None and yverify is not None:
                predictions = self.predict(xverify)
                accuracy = np.sum(yverify == predictions) / float(len(yverify))
                print '  epoch {0}, validation accuracy {1:.4%}, cost: {2}'.format(
                    epoch+1, accuracy, self.cost.eval({self.x: xdata, self.y: ydata})
                    )
            if xverify is not None and yverify is not None and (epoch+1) % validation_check == 0:
                #predictions = self.predict(xverify)
                #accuracy = np.sum(yverify == predictions) / float(len(yverify))
                #print
                #print '  epoch {0}, validation accuracy {1:.4%}'.format(epoch+1, accuracy) 
                if accuracy > max_validation_accuracy:
                    max_validation_accuracy = accuracy
                    best_vars = [x.get_value(borrow=False) for x in self.params]
                elif max_validation_accuracy - accuracy >= early_exit_threshold:
                    print 'Early exit achieved at epoch', epoch
                    break

        # Use the best found weight vectors instead of the end result
        if best_vars is not None:
            print '  Using earlier weight with smaller validation error', max_validation_accuracy
            for i in range(len(self.params)):
                self.params[i].get_value(borrow=True)[:] = best_vars[i]

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
        super(Perceptron, self).__init__(perceptron_loss, dim, 1, r)

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
        super(AveragedPerceptron, self).__init__(perceptron_loss, dim, 1, r)

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


    def train(self, xdata, ydata, epochs, batchSize, xverify=None, yverify=None):
        '''
        Calls the base class train() method and then does post-processing
        '''
        w_avg_before = self.w_avg.get_value(borrow=False)
        b_avg_before = self.b_avg.get_value(borrow=False)
        super(AveragedPerceptron, self).train(xdata, ydata, epochs, batchSize,
            xverify=xverify,
            yverify=yverify,
            )
        self.w_avg = w_avg_before + (self.w_avg - w_avg_before) / len(xdata)
        self.b_avg = b_avg_before + (self.b_avg - b_avg_before) / len(xdata)

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

    def __init__(self, dim, r, C):
        '''
        @param dim Number of dimensions in the input
        @param r   Learning rate
        @param C   First Step
        '''
        self.C = floatX(C)
        self.r0 = floatX(r)
        self.t = theano.shared(floatX(0),  name='t')
        my_loss = lambda x, y, w, b: svm_cost(x, y, w, b, self.C)
        r = theano.shared(self.r0, name='r')
        super(SVM, self).__init__(my_loss, dim, 1, r)
        self.updates.append((self.r, self.r0 / (1 + self.r0 * self.t * self.C)))
        self.updates.append((self.t, self.t + self.x.shape[0]))

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

def _negative_log_likelihood(x, y, w, b, logreg=None):
    '''
    @param x: input
    @param y: labels
    @param w: weight matrix
    @param b: bias vector
    @param logreg: (LogisticRegression object) to save prob
    '''
    # TODO: Fix this.  It runs, but doesn't work for out_dim == 1
    if w.get_value(borrow=True).shape[-1] == 1:
        prob = T.nnet.sigmoid(y*(x.dot(w) + b).transpose())
        nll = -T.log(prob).sum()
    else:
        prob = T.nnet.softmax(x.dot(w) + b)
        nll = -T.log(prob)[T.arange(y.shape[0]), y].sum()
    if logreg is not None:
        logreg.prob = prob
    return nll

def _logreg_cost(x, y, w, b, C, logreg=None):
    return (
        _negative_log_likelihood(x, y, w, b, logreg)
        + C * l2_norm(w)
        )

class LogisticRegression(Sgd):
    def __init__(self, dim_in, dim_out, r, C, x=None):
        self.C = floatX(C)
        self.r0 = floatX(r)
        self.t = theano.shared(floatX(0),  name='t')
        self.prob = None
        r = theano.shared(self.r0, name='r')
        my_cost = lambda x, y, w, b: _logreg_cost(x, y, w, b, self.C, self)
        super(LogisticRegression, self).__init__(my_cost, dim_in, dim_out, r, x)
        self.updates.append((self.r, self.r0 / (1 + self.r0 * self.t * self.C)))
        self.updates.append((self.t, self.t + self.x.shape[0]))

        if dim_out > 1:
            predictor_function = T.argmax(self.prob, axis=1)
        else:
            predictor_function = (T.flatten(T.sgn(self.x.dot(self.w) + self.b)) + 1) / 2

        self._predictor = theano.function(
            inputs=[self.x],
            outputs=predictor_function,
            allow_input_downcast=True,
            name='predictor',
            )

    def predict(self, xdata):
        return self._predictor(xdata)

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
        w_range = floatX(np.sqrt(6. / (dim_in + dim_out)))
        if activation == T.nnet.sigmoid:
            w_range *= 4
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

def mlp_cost(x, y, w, b, C, w_hidden, b_hidden, logreg=None):
    '''
    @param x: input features
    @param y: correct labels of -1 or 1
    @param w: weight vector
    @param b: bias term
    @param C: constant in front of the L2 loss
    @param w_hidden: hidden layer weight matrix
    @param b_hidden: hidden layer bias vector
    '''
    return (
        C * l2_norm(w, w_hidden)
        + _negative_log_likelihood(x, y, w, b, logreg)
        )

class Mlp(Sgd):
    def __init__(self, dim_in, dim_hidden, dim_out, r, C, x=None, activation=T.tanh):
        '''
        @param dim_in: number of input features
        @param dim_hidden: number of nodes in the hidden layer
        @param dim_out: number of outputs
        @param r: learning rate
        @param C: tradeoff between regularizer and loss
        @param x: (theano.tensor.TensorType - one minibatch) symbolic variable
            for the input features.  If none, then it is assumed that this is
            the first layer and a variable will be created.
        @param activation: activation function to use
        '''
        self.hiddenLayer = MlpHiddenLayer(
            dim_in,
            dim_hidden,
            x=x,
            activation=activation,
            )
        self.C = floatX(C)
        self.r0 = floatX(r)
        my_cost = lambda xin, y, w, b: (
            mlp_cost(xin, y, w, b, self.C,
                     self.hiddenLayer.w, self.hiddenLayer.b, self)
            )
        self.t = theano.shared(floatX(0), name='t')
        r = theano.shared(self.r0, name='r')
        super(Mlp, self).__init__(
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
            (self.r, self.r0 / (1 + self.r0 * self.t * self.C)),
            (self.t, self.t + self.x.shape[0]),
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
        answers = theano.function(
            inputs=[self.x],
            outputs=T.argmax(self.prob, axis=1),
            allow_input_downcast=True,
            name='predict',
            )
        return answers(xdata)

class ConvPoolLayer(object):
    '''
    This layer performs convolutions and pooling on input images to create
    output images.
    '''

    def __init__(self, in_shape, in_im_count, out_im_count, filter_shape,
                 pool_shape, batch_size, x=None, activation=T.tanh):
        '''
        @param in_shape: shape of each input image
        @param in_im_count: how many images incoming are to be convolved
            together
        @param out_im_count: how many convolutions to perform in the input,
            each one generating a different output image
        @param filter_shape: shape of the convolution filter.  This should be
            odd integer shapes.
        @param pool_shape: shape of the pooling filter.  This shape should
            evenly divide the in_shape.
        @param x: (theano.tensor.dtensor4) input images to convolve around.
            The dimensions of this input array is
               (batch_size, in_im_count, in_shape[0], in_shape[1])
            If this is None, then this layer is considered as the first layer
            and the layer will create an x variable for you.
        @param activation: activation function to use after pooling.
        '''
        # TODO: Something is broken here.  Exception raised in training
        self.in_shape = in_shape
        self.in_im_count = in_im_count
        self.out_im_count = out_im_count
        self.filter_shape = filter_shape
        mid_shape = (
            in_shape[0] - filter_shape[0] + 1,
            in_shape[1] - filter_shape[1] + 1,
            )
        self.out_shape = (
            mid_shape[0] / pool_shape[0],
            mid_shape[1] / pool_shape[1],
            )

        if x is None:
            self.x = T.matrix('x')
            self.x = self.x.reshape((batch_size, in_im_count, in_shape[0], in_shape[1]))
        else:
            self.x = x

        fan_in = in_im_count * np.prod(filter_shape)
        fan_out = out_im_count * np.prod(filter_shape) / np.prod(pool_shape)
        w_range = np.sqrt(6.0 / (fan_in + fan_out))
        if activation == T.nnet.sigmoid:
            w_range *= 4
        self.w = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    low=-w_range,
                    high=w_range,
                    size=(out_im_count, in_im_count, filter_shape[0], filter_shape[1]),
                    ),
                dtype=theano.config.floatX
                ),
            name='w',
            borrow=True
            )
        self.b = theano.shared(
            value=np.zeros((out_im_count,), dtype=theano.config.floatX),
            name='b',
            borrow=True
            )

        conv_out = T.nnet.conv.conv2d(
            input=self.x,
            filters=self.w,
            image_shape=(None, in_im_count, in_shape[0], in_shape[1]),
            filter_shape=(out_im_count, in_im_count, filter_shape[0], filter_shape[1]),
            )

        pool_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_shape,
            ignore_border=True
            )

        self.activation = activation
        self.output = activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.w, self.b]

class ConvNet(Mlp):
    '''
    Creates a potentially multi-layered convolutional neural network with an
    MLP at the end of the chain.
    '''

    def __init__(self, in_shape, out_im_counts, filter_shapes, pool_shapes,
                 batch_size, dim_hidden, dim_out, r, C, x=None,
                 activation=T.tanh):
        '''
        @param in_shape: shape of each input image
        @param out_im_counts: tuple of number of output images at each conv layer
        @param filter_shapes: tuple of tuples, filter shape at each conv layer
        @param pool_shapes: tuple of tuples, pool shape at each conv layer
        @param batch_size: size of batches
        @param x: (theano.tensor.dtensor4) input images to convolve around.
            The dimensions of this input array is
               ('x', in_im_count, in_shape[0], in_shape[1])
            where 'x' represents a` broadcastable dimension (i.e. for mini-batch
            gradient descent).  If this is None, then this layer is considered
            as the first layer and the layer will create an x variable for you.
        @param activation: activation function to use in the whole network
        '''
        conv_layers = [
            ConvPoolLayer(in_shape, 1, out_im_counts[0], filter_shapes[0],
                          pool_shapes[0], batch_size, x=x,
                          activation=activation)
            ]
        for i in range(1, len(out_im_counts)):
            prev_layer = conv_layers[i-1]
            conv_layers.append(ConvPoolLayer(
                prev_layer.out_shape,
                prev_layer.out_im_count,
                out_im_counts[i],
                filter_shapes[i],
                pool_shapes[i],
                batch_size,
                x=prev_layer.output,
                activation=activation
                ))

        mlp_input = conv_layers[-1].output.flatten(1)
        super(ConvNet, self).__init__(
            np.prod(conv_layers[-1].out_shape),
            dim_hidden,
            dim_out,
            r,
            C,
            x=mlp_input,
            activation=activation
            )
        self.x = conv_layers[0].x  # Reset x to first input

        self.params.extend([x for layer in conv_layers for x in layer.params])
        self.updates.extend([
            (param, param - self.r * T.grad(self.cost, wrt=param))
            for layer in conv_layers for param in layer.params
            ])

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
    parser.add_argument('-c', '--classifier', default='SVM', help='''
        Choices are Perceptron, AveragedPerceptron, SVM, LogisticRegression,
        and MLP.
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
        #('train1.10', 'data1/train1.10', 'data1/test1.10'),
        ('astro-original', 'astro/original/train', 'astro/original/test'),
        ('astro-scaled', 'astro/scaled/train', 'astro/scaled/test'),
        ]

    for name, trainingPath, testingPath in sets:
        training = TrainingExample.fromSvm(
            os.path.join(args.base_dir, trainingPath)
            )
        testing = TrainingExample.fromSvm(
            os.path.join(args.base_dir, testingPath)
            )
        testLearner(name, training, testing,
                    crossepochs=10,
                    epochs=epochs,
                    batchSize=batchSize,
                    learnerName=args.classifier)
    print

def testLearner(name, trainExamples, testExamples, crossepochs=10,
                epochs=10, batchSize=1, learnerName='SVM'):
    '''
    Trains an averaged Perceptron classifier from the training examples and
    then calculates the accuracy of the generated classifier on the test
    examples.

    Prints out the results to the console
    '''
    featuresList = np.asarray([x.features for x in trainExamples], dtype=theano.config.floatX)
    labels = np.asarray([x.label for x in trainExamples], dtype=theano.config.floatX)
    testFeatures = np.asarray([x.features for x in testExamples], dtype=theano.config.floatX)
    testLabels = np.asarray([x.label for x in testExamples], dtype=theano.config.floatX)

    rvalues = [0.01, 0.05, 0.1, 0.5]
    Cvalues = [0.001, 0.005, 0.01, 0.05]
    dimvalues = [10, 20]

    logreglearner = lambda dim_in, r, C: LogisticRegression(dim_in, 2, r, C)
    mlplearner = lambda dim_in, dim_hidden, r, C: Mlp(dim_in, dim_hidden, 2, r, C)
    learnerMap = {
        'Perceptron': (Perceptron, [(x,) for x in rvalues], ['r']),
        'AveragedPerceptron': (AveragedPerceptron, [(x,) for x in rvalues], ['r']),
        'SVM': (SVM, list(itertools.product(rvalues, Cvalues)), ['r', 'C']),
        'LogisticRegression': (
            logreglearner,
            list(itertools.product(rvalues, Cvalues)),
            ['r', 'C']
            ),
        'MLP': (
            mlplearner,
            list(itertools.product(dimvalues, rvalues, Cvalues)),
            ['hidden-dimension', 'r', 'C']
            ),
        }
    learner, hypers, names = learnerMap[learnerName]
    
    logisticLearners = ('LogisticRegression', 'MLP')
    if learnerName in logisticLearners:
        labels -= labels.min()
        labels /= labels.max()
        testLabels -= testLabels.min()
        testLabels /= testLabels.max()

    k = 5
    print 'Performing cross-validation'
    print '  dataset:        ', name
    print '  learner:        ', learnerName
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

