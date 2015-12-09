import TrainingExample

import numpy as np
import random
import theano
import theano.tensor as T

class Sgd(object):
    def __init__(self, cost_gen, dim_in, dim_out, r):
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
        '''
        self.r = r

        self.x = T.matrix('x')
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

        updates = [
            (self.w, self.w - self.r * self.dcost_dw),
            (self.b, self.b - self.r * self.dcost_db),
            ]

        index = T.lscalar()
        trainingFunction = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=updates,
            givens={
                self.x: xdata_share[index*batchSize : (index+1)*batchSize],
                self.y: ydata_share[index*batchSize : (index+1)*batchSize],
                }
            )

        # TODO: permute xdata and ydata
        for epoch in xrange(epochs):
            #perm = np.random.permutation(xlen)
            #xdata[perm] = xdata
            #ydata[perm] = ydata
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
                #print self.w.get_value(borrow=True).reshape(len(xdata[0]))
                trainingFunction(minibatchIndex)

def perceptron_loss(x, y, w, b):
    '''
    Returns a perceptron loss function

    @param x  Features (theano matrix)
    @param y  Label (theano ivector)
    @param w  Weight matrix (theano matrix)
    @param b  Bias vector (theano vector)
    @return The perceptron loss function with x, y, w, and b inside
    '''
    return T.sum(T.maximum(0, - y * (x.dot(w) + b)))

class Perceptron(Sgd):
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
        # The answers array is shaped as a (n,1) 2D array.  We want to reshape to a 1D array.
        return answers.reshape((xdata_share.get_value(borrow=True).shape[0],))

def main():
    r = 0.2
    print 'learning rate fixed:   r =', r

    sanityExamples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/sanityCheck-train.dat')
    print 'sanity train accuracy:    ', testPerceptron(sanityExamples, sanityExamples, r)

    train10Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data0/train0.10')
    test10Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data0/test0.10')
    print 'train0.10 train accuracy: ', testPerceptron(train10Examples, train10Examples, r)
    print 'train0.10 test accuracy:  ', testPerceptron(train10Examples, test10Examples, r)

    train20Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data0/train0.20')
    test20Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data0/test0.20')
    print 'train0.20 train accuracy: ', testPerceptron(train20Examples, train20Examples, r)
    print 'train0.20 test accuracy:  ', testPerceptron(train20Examples, test20Examples, r)

    train110Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data1/train1.10')
    test110Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data1/test1.10')
    print 'train1.10 train accuracy: ', testPerceptron(train110Examples, train110Examples, r)
    print 'train1.10 test accuracy:  ', testPerceptron(train110Examples, test110Examples, r)

    train120Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data1/train1.20')
    test120Examples = TrainingExample.fromSvm('/home/bentley/classes/cs6350_machine_learning/hw/handin/hw02/data/data1/test1.20')
    print 'train1.20 train accuracy: ', testPerceptron(train120Examples, train120Examples, r)
    print 'train1.20 test accuracy:  ', testPerceptron(train120Examples, test120Examples, r)


def testPerceptron(trainExamples, testExamples, r = 0.2):
    '''
    Trains a Perceptron classifier from the training examples and then
    calculates the accuracy of the generated classifier on the test examples.
    Returns a ratio of correct test examples over total test examples.
    '''
    featuresList = [x.features for x in trainExamples]
    labels = [x.label for x in trainExamples]
    p = Perceptron(len(featuresList[0]), r)
    p.train(featuresList, labels, 10, 1)
    testFeatures = [x.features for x in testExamples]
    testLabels = [x.label for x in testExamples]
    predictions = p.predict(testFeatures)
    #print predictions
    return np.sum(testLabels == predictions) / float(len(testLabels))
    #print p.w.get_value()
    #print p.predict(featuresList)

if __name__ == '__main__':
    main()

