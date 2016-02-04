'''
Script for using a trained classifier from train.py
'''

from imagefuncs import loadImage
from train import preprocessData
from learners import ConvNet

import argparse
import cPickle
import csv
import os
import sys

import numpy as np

def parseArgs(arguments):
    'Parse command-line arguments'
    parser = argparse.ArgumentParser(description='''
        Performs classification on images using a previously trained
        classifier.  These images need to be in the same format used by
        create_image_cache.py.  More specifically, the images must be 68x40
        pixels in size.  If the original image is smaller than this, then you
        will want to repeat the last row or column on each side of the image to
        extend it to 68x40 (this is the way create_image_cache.py does it).
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', help='directory containing images to be classified')
    parser.add_argument('-o', '--output', default='classified-labels.tsv', help='''
        classification output with columns "imagepath" and "classification"
        where classification will be "M" or "F"
        ''')
    parser.add_argument('-c', '--classifier', default='classifier.dat', help='''
        classifier data file generated from train.py
        ''')
    return parser.parse_args(args=arguments)

def main(arguments):
    'Main entry point'
    args = parseArgs(arguments)
    with open(args.classifier, 'rb') as classifierFile:
        classifier = cPickle.load(classifierFile)
    xdata = []
    imagepaths = []
    print 'Reading in images...'
    for entry in os.listdir(args.input):
        filepath = os.path.join(args.input, entry)
        if os.path.isfile(filepath):
            try:
                xdata.append(loadImage(filepath))
            except IOError:
                # This probably wasn't an image file or bad permissions
                print >> sys.stderr, 'Warning: Could not read image', filepath
            else:
                # Only store the filepath if we were able to load the image
                imagepaths.append(filepath)
    print '  done'

    print 'Preprocessing data...'
    xdata = np.asarray(xdata)
    xdata = preprocessData(xdata, reshape=(not isinstance(classifier, ConvNet)))
    print '  done'

    print 'Classifying the data...'
    labels = classifier.predict(xdata)
    print '  done'

    # Output the results to a file
    with open(args.output, 'w') as outfile:
        writer = csv.writer(outfile, dialect='excel-tab')
        writer.writerow(['imagepath', 'classification'])
        for imagepath, label in zip(imagepaths, labels):
            labelString = 'M' if label == 1 else 'F'
            writer.writerow([imagepath, labelString])
    print 'Wrote classification data to', args.output

if __name__ == '__main__':
    main(sys.argv[1:])
