'''
Creates an image cache.  Call this script with --help for more info
'''

import imagefuncs
import ldsImporter
import utils

import cv2

import argparse
import csv
import os
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

def main(arguments):
    '''
    Main entry point.  Call with the --help option for more information
    on usage.
    '''
    args = parseArgs(arguments)
    with open(args.filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')
        lines = [line for line in reader]
    imagepaths = set(line['imagePath'] for line in lines)
    print len(imagepaths), 'images'
    utils.mkdir_p(args.outdir)
    newsize = (40, 68)
    for imagepath in imagepaths:
        image = cv2.imread(imagepath)
        relevantLines = [
            line for line in lines if line['imagePath'] == imagepath
            ]
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
                args.outdir,
                imagenameComponents[0] + '-' + line['line'] + '.png'
                )
            cv2.imwrite(outpath, newimage)
        print imagepath

if __name__ == '__main__':
    main(sys.argv[1:])

