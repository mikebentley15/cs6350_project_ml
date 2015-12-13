'''
Creates an image cache.  Call this script with --help for more info
'''

import imagefuncs
import ldsimporter
import utils

import cv2
import numpy as np

import argparse
import csv
import os
import pickle
import sys

def parseArgs(arguments):
    '''
    Parses the arguments for this script.  Returns a parser arguments object.
    '''
    parser = argparse.ArgumentParser(description='''
        Reads data from a tsv file, including image path, and generates an
        image cache for the 'sex' field of the images.   The images output will
        be all the same size, which is 40 x 68.  The names of the images
        in the cache directory will be <input-image-name>-<line-number>.png.
        ''')
    parser.add_argument('filepath', help='tsv file containing labeled data')
    parser.add_argument('outdir', help='Where to generate the image cache')
    return parser.parse_args(args=arguments)

def separateImages(filepath, outdir):
    '''
    Separates the images of the gender field of all entries in the given tsv
    file.
    '''
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')
        lines = [line for line in reader]
    imagepaths = set(line['imagePath'] for line in lines)
    print len(imagepaths), 'images'
    utils.mkdir_p(outdir)
    newsize = (40, 68)
    for imagepath in imagepaths:
        image = cv2.imread(imagepath)
        if image is None:
            print imagepath
            raise IOError(imagepath + ' does not exist')
        relevantLines = [
            line for line in lines if line['imagePath'] == imagepath
            ]
        for line in relevantLines:
            boundbox = ldsimporter.BoundingBox(
                int(line['sex-top']),
                int(line['sex-bottom']),
                int(line['sex-left']),
                int(line['sex-right'])
                )
            newimage = imagefuncs.getSubImage(image, boundbox)
            newimage = imagefuncs.mirrorPadImage(newimage, newsize)
            imagenameComponents = os.path.splitext(os.path.basename(imagepath))
            outpath = os.path.join(
                outdir,
                imagenameComponents[0] + '-' + line['line'] + '.png'
                )
            cv2.imwrite(outpath, newimage)
        print imagepath

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
        for line in reader:
            filename = os.path.basename(line['imagePath'])
            split = os.path.splitext(filename)
            cachepic = split[0] + '-' + line['line'] + '.png'
            xdata.append(imagefuncs.loadImage(os.path.join(cachedir, cachepic)))
            ydata.append(1 if line['truth-sex'] == 'M' else -1)
    return np.asarray(xdata), np.asarray(ydata)

def createCache(filepath, outdir):
    '''
    Creates a cache.pkl file containing the xdata and ydata numpy arrays.
    >>> createCache(filepath, outdir)
    >>> xdata, ydata = pickle.load(open(os.path.join(outdir, 'cache.pkl')))
    '''
    print 'Loading records from file'
    xdata, ydata = loadrecords(filepath, outdir)
    cachepath = os.path.join(outdir, 'cache.pkl')
    print 'Pickling a cache file: ', cachepath
    with open(cachepath, 'w') as cachefile:
        pickle.dump((xdata, ydata), cachefile)

def main(arguments):
    '''
    Main entry point.  Call with the --help option for more information
    on usage.
    '''
    args = parseArgs(arguments)
    separateImages(args.filepath, args.outdir)
    createCache(args.filepath, args.outdir)

if __name__ == '__main__':
    main(sys.argv[1:])

