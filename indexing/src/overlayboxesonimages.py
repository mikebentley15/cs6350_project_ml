from ldsimporter import BoundingBox
from utils import mkdir_p
from imagefuncs import drawBox

import argparse
import csv
import os
import sys
import cv2 # OpenCV

_defaultOutDir = 'output'

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        This script will add black boxes where the bounding boxes are
        defined in the csv file.  It will output these copies to the
        outdir directory.  Note that if outdir == indir, then the
        images will be overwritten with the versions that have the bounding
        boxes overlayed.
        
        The images are expected to be found relative to the current
        directory using the path in the csv file.  Unless the csv file
        has absolute paths...
        ''')
    parser.add_argument('-o', '--outdir', default=_defaultOutDir, help='''
        output directory for image copies
        ''')
    parser.add_argument('csv', help='''
        CSV file containing the data.
        ''')
    args = parser.parse_args(args=arguments)
    mkdir_p(args.outdir)
    with open(args.csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')
        curImPath = None
        im = None
        for line in reader:
            if curImPath != line['imagePath']:
                if im is not None:
                    cv2.imwrite(os.path.join(args.outdir, os.path.basename(curImPath)), im)
                curImPath = line['imagePath']
                im = cv2.imread(curImPath) #, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                print curImPath
            boxes = []
            for boxtype in ('sex', 'race', 'married'):
                boxes.append(BoundingBox(
                    int(line[boxtype + '-top']),
                    int(line[boxtype + '-bottom']),
                    int(line[boxtype + '-left']),
                    int(line[boxtype + '-right']),
                    ))
            for box in boxes:
                drawBox(im, box, [0, 0, 255]) # Draw the box red
        if im is not None:
            cv2.imwrite(os.path.join(args.outdir, os.path.basename(curImPath)), im)

if __name__ == '__main__':
    main(sys.argv[1:])

