import os
import sys
import timeit
import imagefuncs
import ldsImporter
import utils

import argparse

import numpy
import csv

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cv2

def main(filepath, outdir):
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')
        lines = [line for line in reader]
    imagepaths = set(line['imagePath'] for line in lines)
    print len(imagepaths), 'images'
    utils.mkdir_p(outdir)
    newsize = (52, 68)
    for imagepath in imagepaths:
        image = cv2.imread(imagepath)
        relevantLines = [line for line in lines if line['imagePath'] == imagepath]
        for line in relevantLines:
            boundbox = ldsImporter.BoundingBox(
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('outdir')
    args = parser.parse_args()
    main(args.filepath, args.outdir)

