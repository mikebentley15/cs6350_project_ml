'''
Provides a TrainingExample named tuple for convenience.

Also, factory functions exist such as trainingExamplesFromSvm().
'''

from collections import namedtuple
import numpy

TrainingExample = namedtuple('TrainingExample', ('label', 'features'))
# '''
# TrainingExample: NamedTuple to contain a single training example
# 
# Attributes:
#     label:    Label attached to the label, can be of any type you need, usually str
#     features: A numpy array of features and values.  Note: not a sparse array.
# '''

def fromSvm(filepath):
    '''
    Loads the training examples from the given file in libSVM format.

    Returns a list of training examples, one for each row.
    '''
    labels = []
    sparseFeatureList = []
    with open(filepath, 'r') as trainFile:
        for line in trainFile:
            split = line.split()
            labels.append(int(split[0]))
            sparseFeatureList.append(sorted([
                (int(x.split(':')[0]), float(x.split(':')[1]))
                for x in split[1:]
                ]))
    maximumIndex = max(x[-1][0] for x in sparseFeatureList)
    trainingExamples = []
    for trainingIndex, sparseFeatures in enumerate(sparseFeatureList):
        features = numpy.zeros(maximumIndex + 1)
        for featureIndex, value in sparseFeatures:
            features[featureIndex] = value
        trainingExamples.append(TrainingExample(labels[trainingIndex], features))
    return trainingExamples