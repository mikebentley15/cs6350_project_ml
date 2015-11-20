'''
Converts the xml information from the data files for the church
into an intermediate format which is a single tsv file for the
whole lot.
'''

import ldsImporter

import argparse
import csv
import errno
import os
import sys

_scriptDir = os.path.dirname(__file__)
_defaultTrainDir = os.path.join(_scriptDir, '..', 'training-data')
_defaultTestDir = os.path.join(_scriptDir, '..', 'test-data')
_defaultOutDir = 'output'

def mkdir_p(path):
    '''
    Performs the same functionality as mkdir -p in the bash shell.
    An OSError is raised if the directory does not exist and was
    not able to be created.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main(arguments):
    'Main entry point'
    parser = argparse.ArgumentParser(description='''
        This script will parse the images and xml files in the provided
        training data directory and will train some classifiers.  Those
        classifiers will be output to data files in the output directory.
        Both the training data and test data will be classified using the
        generated classifiers and will be output in subdirectories of the
        output directory.
        ''')
    parser.add_argument('--train', default=_defaultTrainDir, help='''
        Directory containing the training set.
        This directory is expected to have the following files:
        <image-name>.jpg,
        <image-name>.truth.xml,
        <image-name>.origABARB.xml,
        <image-name>.hypBboxes.xml.filtered
        ''')
    parser.add_argument('--test', default=_defaultTestDir, help='''
        Directory containing the test set.
        This directory is expected to have the same file structure as the train directory.
        ''')
    parser.add_argument('-o', '--outdir', default=_defaultOutDir, help='''
        Where to output the results.
        The structure of the output directory will be in two files,
        train.tsv and test.tsv.
        ''')
    args = parser.parse_args(args=arguments)
    trainData = ldsImporter.readFiles(args.train)
    testData = ldsImporter.readFiles(args.test)
    mkdir_p(args.outdir)
    for data, outfile in ((trainData, os.path.join(args.outdir, 'train.tsv')),
                          (testData, os.path.join(args.outdir, 'test.tsv'))):
        with open(outfile, 'w') as outfileObj:
            writer = csv.writer(outfileObj, dialect='excel-tab')
            headerRow = ['imagePath', 'line']
            for name in ('truth', 'a', 'b', 'arb', 'company'):
                headerRow.extend(
                    name + '-' + extra for extra in ('sex', 'race', 'married')
                    )
            for name in ('sex', 'race', 'married'):
                headerRow.extend(
                    name + '-' + coord
                    for coord in ('top', 'bottom', 'left', 'right')
                    )
            writer.writerow(headerRow)
            for imageData in data:
                count = len(imageData.trueRecords)
                assert count == len(imageData.aRecords), \
                    'aRecords count mistmatch: {0} != {1}, file={2}'.format(count, len(imageData.aRecords), imageData.imagePath)
                assert count == len(imageData.bRecords), \
                    'bRecords count mistmatch: {0} != {1}, file={2}'.format(count, len(imageData.bRecords), imageData.imagePath)
                assert count == len(imageData.arbRecords), \
                    'arbRecords count mistmatch: {0} != {1}, file={2}'.format(count, len(imageData.arbRecords), imageData.imagePath)
                assert count <= len(imageData.companyRecords), \
                    'companyRecords count mistmatch: {0} !<= {1}, file={2}'.format(count, len(imageData.companyRecords), imageData.imagePath)
                assert count <= len(imageData.boundingBoxes), \
                    'boundingBoxes count mistmatch: {0} !<= {1}, file={2}'.format(count, len(imageData.boundingBoxes), imageData.imagePath)
                lines = set(x.line for x in imageData.trueRecords)
                # Make a row in the csv file per line
                for line in sorted(lines):
                    row = [imageData.imagePath, line]
                    for name, records in (('truth', imageData.trueRecords),
                                          ('a', imageData.aRecords),
                                          ('b', imageData.bRecords),
                                          ('arb', imageData.arbRecords),
                                          ('company', imageData.companyRecords)):
                        # Note: the mod 50 is because there are at most 50 records in each image
                        #  and at least the following image has renumbered the rows, where company
                        #  data was confused:
                        #   - 004531795_01081.jpg
                        element = [x for x in records if (x.line % 50) == (line % 50)]
                        # There should be exactly one record from each source for each line
                        assert len(element) == 1, 'Type {0} has {1} records for line {2}, file = {3}'.format(name, len(element), line, imageData.imagePath)
                        row.extend(element[0][1:]) # Appends sex, race, and married
                    # Note: the mod 50 is for the same reason above
                    boxes = [x for x in imageData.boundingBoxes if (x.line % 50) == (line % 50)]
                    assert len(boxes) == 1
                    for box in boxes[0][1:]:
                        row.extend(box) # Appends top, bottom, left, and right
                    writer.writerow(row)

if __name__ == '__main__':
    main(sys.argv[1:])

