"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit
import itertools
import cPickle
import gzip

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from crossvalidate import crossvalidate
from outdup import OutDuplicator


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        self.out_shape = (
            image_shape[0],
            filter_shape[0],
            (image_shape[2] + 1 - filter_shape[2]) / poolsize[0],
            (image_shape[3] + 1 - filter_shape[3]) / poolsize[1],
            )

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # Michael Q: Why does fan_in depend on number of input images?
        # Michael A: The fan_in depends on the number of input images because
        # Michael A: each output image will be computed from the entire group
        # Michael A: of input images, so consider the input as a 3D image and
        # Michael A: we are generating k output 2D images from the 3D image
        # Michael A: input.  Therefore, this 3D image input is # of input nodes.
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # Michael Q: Why does fan_out depend on the number of output images?
        # Michael A: ?
        # Michael A: I'm not too sure yet.
        # Michael A: Basically in the paper describing the W_bound, it uses
        # Michael A: just the sum of the number of inputs and the number of
        # Michael A: nodes.  I'm not sure why this is necessary, but it may
        # Michael A: have to do with the fact that the outputs are not
        # Michael A: necessarily independent because they are joined in later
        # Michael A: layer(s) of the network, and are connected when doing
        # Michael A: back-propogation.
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        # Michael Q: Where does the weight vector bound come from?
        # Michael A: This particular bound comes from the use of tanh(x) as
        # Michael A: our activation function.  This bound for the tanh(x)
        # Michael A: comes from the paper "Understanding the difficulty of
        # Michael A: training deep feedforward neural networks" by Xavier
        # Michael A: Glorot and Yoshua Bengio.
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        # Michael Q: Why is the b vector simply initialized to zero?
        # Michael A: Not fully sure.
        # Michael A: I think it's because the tanh function is centered about
        # Michael A: the origin.  It may depend on the centering of the input
        # Michael A: values.
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class Trainer(object):
    def __init__(self, r, kerns1, kerns2, batch_size):
        self.r = r
        self.kerns1 = kerns1
        self.kerns2 = kerns2
        self.batch_size = batch_size
        #print 'r    ', r
        #print 'k1   ', kerns1
        #print 'k2   ', kerns2
        #print 'batch', batch_size

        rng = numpy.random.RandomState(23455)

        # These are the dimensions of the images to be processed
        xdim = 40
        ydim = 68

        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        #print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = self.x.reshape((batch_size, 1, xdim, ydim))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, kerns1, 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, xdim, ydim),
            filter_shape=(kerns1, 1, 5, 5),
            poolsize=(4, 4)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, kerns2, 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=layer0.out_shape,
            filter_shape=(kerns2, kerns1, 5, 5),
            poolsize=(4, 4)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, kerns2 * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=numpy.prod(layer1.out_shape[1:]),
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
        self._final_layer = layer3

        # the cost we minimize during training is the NLL of the model
        self.cost = layer3.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        
        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        self.updates = [
            (param_i, param_i - self.r * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

    def train(self, train_set_x, train_set_y, epochs, batch_size,
              valid_set_x=None, valid_set_y=None, quiet=True):
        assert self.batch_size == batch_size

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.shape[0]
        n_train_batches /= batch_size

        xtrain_shr = theano.shared(train_set_x, borrow=True)
        ytrain_shr = theano.shared(train_set_y, borrow=True)


        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # create a function to compute the mistakes that are made by the model
        validate_model = None
        if valid_set_x is not None and valid_set_y is not None:
            n_valid_batches = valid_set_x.shape[0]
            n_valid_batches /= batch_size
            xval_shr = theano.shared(valid_set_x, borrow=True)
            yval_shr = theano.shared(valid_set_y, borrow=True)

            validate_model = theano.function(
                [index],
                self._final_layer.errors(self.y),
                givens={
                    self.x: xval_shr[index * batch_size: (index + 1) * batch_size],
                    self.y: yval_shr[index * batch_size: (index + 1) * batch_size]
                }
            )

        train_model = theano.function(
            [index],
            self.cost,
            updates=self.updates,
            givens={
                self.x: xtrain_shr[index * batch_size: (index + 1) * batch_size],
                self.y: ytrain_shr[index * batch_size: (index + 1) * batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############
        # early-stopping parameters
        patience = 500000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / (2*batch_size))
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < epochs) and (not done_looping):
            epoch = epoch + 1
            print '    epoch {0:3d}:   '.format(epoch),
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                cost_ij = train_model(minibatch_index)
                if (minibatch_index+1) % min(100, (n_train_batches+9)/10) == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if validate_model is not None and (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    if not quiet:
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                              (epoch, minibatch_index + 1, n_train_batches,
                               this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * batch_size * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter * batch_size:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        if not quiet:
            print('Optimization complete.')
            print('Best validation score of %f %% obtained at iteration %i, '
                  'with test performance %f %%' %
                  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
            print >> sys.stderr, ('The code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def errors(self, xdata, ydata):
        n_batches = ydata.shape[0]
        n_batches /= self.batch_size
        xdata_shr = theano.shared(xdata, borrow=True)
        ydata_shr = theano.shared(ydata, borrow=True)

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        test_model = theano.function(
            [index],
            self._final_layer.errors(self.y),
            givens={
                self.x: xdata_shr[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: ydata_shr[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        return numpy.mean([test_model(i) for i in xrange(n_batches)])

    def predict(self, test_set_x):
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        n_test_batches = test_set_x.shape[0]
        n_test_batches /= self.batch_size

        xtest_shr = theano.shared(test_set_x, borrow=True)

        predict_model = theano.function(
            inputs=[index],
            outputs=self._final_layer.y_pred,
            givens={
                self.x: xtest_shr[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )

        answers = numpy.concatenate([
            predict_model(i) for i in xrange(n_test_batches)
            ])
        return answers

def main():
    dataset = '../subimages/cache.pkl.gz'
    datasets = load_data(dataset)
    epochs = 200
    batch_size=500
    rvals = [0.1, 0.05, 0.01, 0.005]
    kerns1vals = [10, 20]
    kerns2vals = [20, 50]
    hypers = list(itertools.product(rvals, kerns1vals, kerns2vals))
    names = ['r', 'k1', 'k2']
    trainerWrapper = lambda dim_in, r, k1, k2: Trainer(r, k1, k2, batch_size)

    xdata, ydata = datasets[0]
    xverify, yverify = datasets[1]
    xtest, ytest = datasets[2]

    xdata_ref = xdata.get_value(borrow=True)
    ydata_ref = ydata.eval()
    xverify_ref = xverify.get_value(borrow=True)
    yverify_ref = yverify.eval()
    xtest_ref = xtest.get_value(borrow=True)
    ytest_ref = ytest.eval()

    print 'Learning a convolutional neural network (CNN)'
    print '  dataset:            ', dataset
    print '  training size:      ', xdata_ref.shape
    print '  verify size:        ', xverify_ref.shape
    print '  test size:          ', xtest_ref.shape

    k = 3
    cross_epochs = 5
    print '... cross-validating'
    print '  k:                    ', k
    print '  cross-val epochs:     ', cross_epochs
    print '  batch size:           ', batch_size
    print '  params:               ', names
    for i in xrange(len(names)):
        print '  {0:3s} values:         '.format(names[i]), sorted(set([x[i] for x in hypers]))
    #best = crossvalidate(trainerWrapper, xdata_ref, ydata_ref, k, cross_epochs, batch_size, hypers, names)
    print 'Skipping cross-validation.  Using best value from previous run'
    best = [0.05, 20, 50]
    print '  best hyper-params:    ', names, ' = ', best

    classifier = trainerWrapper(0, *best)
    print '... training'
    print '  batch size:           ', batch_size
    print '  epochs:               ', epochs
    print '  hyper-params:         ', names, ' = ', best
    classifier.train(xdata_ref, ydata_ref, epochs, batch_size, valid_set_x=xverify_ref,
                     valid_set_y=yverify_ref, quiet=False)
    print '... testing'
    print '  training accuracy:    ', 1 - classifier.errors(xdata_ref, ydata_ref)
    print '  validation accuracy:  ', 1 - classifier.errors(xverify_ref, yverify_ref)
    print '  testing accuracy:     ', 1 - classifier.errors(xtest_ref, ytest_ref)
    print '... pickling'
    print '  classifier:           ', classifier
    filename = pickleSafely(classifier, 'classifier', '.pkl.gz')
    print '  saved to:             ', filename

def pickleSafely(toPickle, fileBase, ext):
    i = 1
    namer = lambda base, i, ext: '{0}-{1:02d}{2}'.format(base, i, ext)
    filename = namer(fileBase, i, ext)
    while os.path.exists(filename):
        i += 1
        filename = namer(fileBase, i, ext)
    with gzip.open(filename, 'wb') as zipfile:
        cPickle.dump(toPickle, zipfile)
    return filename

if __name__ == '__main__':
    logfile = 'console.log'
    i = 0
    while os.path.exists(logfile):
        i += 1
        logfile = 'console-{0:02d}.log'.format(i)
    with open(logfile, 'w') as outfile:
        origstdout = sys.stdout
        try:
            sys.stdout = OutDuplicator([origstdout, outfile])
            main()
        finally:
            sys.stdout = origstdout

def experiment(state, channel):
    evaluate_lenet5(load_data(state.dataset), state.learning_rate)

