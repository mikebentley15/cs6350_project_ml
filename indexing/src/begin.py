import os
import sys
import timeit
import imagefuncs
import ldsImporter
import utils

import numpy
import csv

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cv2

#   SubImage:
#   outpath    (str) Path of the subimage file
#   truth-sex  (str) Label of the subimage
SubImage = namedtuple('SubImage', 'outpath truth-sex')

data=[]
with open(filepath, 'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='excel-tab')
    for line in reader:
        imagepath = os.path.join('../../data', os.path.basename(line['imagePath']))
        image = cv2.imread(imagepath)
        boundbox = ldsImporter.BoundingBox(
            int(line['sex-top']),
            int(line['sex-bottom']),
            int(line['sex-left']),
            int(line['sex-right'])
            )
        newimage = imagefuncs.getSubImage(image, boundbox)
        outdir = '../../subimages'
        utils.mkdir_p(outdir)
        imagenameComponents = os.path.splitext(os.path.basename(imagepath))
        outpath = os.path.join(
            outdir,
            imagenameComponents[0] + '-' + line['line'] + '.' + imagenameComponents[1]
            )
        cv2.imwrite(outpath, newimage)
        #data.append(SubImage(outpath,line['truth-sex'])
        #I want to make an object the object will include..... outpath,truth-sex 


#datasets = load_data(dataset)
#train_set = datasets[0]
#test_set = datasets[1]

        
