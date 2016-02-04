#!/usr/bin/env python

from PIL import Image
import csv
import os
import sys

scriptdir = os.path.dirname(__file__)

datadir = os.path.join(scriptdir, 'data')
datafiles = [
    os.path.join(datadir, 'train.csv'),
    os.path.join(datadir, 'test.csv'),
    ]
outdir = os.path.join(scriptdir, 'output')

def main():
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for f in datafiles:
        fbase = os.path.splitext(os.path.basename(f))[0]
        with open(f, 'r') as infile:
            reader = csv.reader(infile)
            reader.next()
            for idx, row in enumerate(reader):
                im = Image.new('L', (28, 28))
                endIdx = len(row) - 1
                startIdx = endIdx + 1 - 28*28
                im.putdata([int(x) for x in row[startIdx:endIdx]])
                outname = os.path.join(outdir, fbase + '{:05}'.format(idx) + '.png')
                im.save(outname)
                print outname


if __name__ == '__main__':
    main()
