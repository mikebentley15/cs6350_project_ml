from collections import namedtuple

# BoundingBox:
#   top     (int) Top of the box
#   bottom  (int) Bottom of the box
#   left    (int) Left of the box
#   right   (int) Right of the box
BoundingBox = namedtuple('BoundingBox', 'top bottom left right')

# Record:
#   line    (int) Line number on the image for the record
#   sex     (string) Gender - either 'M', 'F', or ''
#   race    (string) Race or color - e.g. 'White'
#   married (string) Martial status - either 'M', 'S', 'D', 'W', or ''
Record = namedtuple('Record', 'line sex race married')

# RecordBoundingBoxes
#   line        (int) Line number on the image for the record
#   sexBox      (BoundingBox) Bounding box for the sex field on the record's line
#   raceBox     (BoundingBox) Bounding box for the race field on the record's line
#   marriedBox  (BoundingBox) Bounding box for the marital status field on the record's line
RecordBoundingBoxes = namedtuple('RecordBoundingBoxes', 'line sexBox raceBox marriedBox')

def getAttributeContents(node, attribute):
    '''
    Returns the content of the given attribute to the given node.

    If the attribute doesn't exist (or there are more than one), then an
    AssertionError is thrown.
    '''
    a = node.xpathEval('./@' + attribute)
    assert len(a) > 0, 'Attribute {0} not found in node {1}'.format(attribute, node)
    assert len(a) < 2, 'Duplicate attributes {0} found in node {1}'.format(attribute, node)
    return a[0].get_content()

class ImageData(object):
    '''
    Represents the data for a single image
    Has the following attributes:
    - imagePath       File path to image file
    - trueRecords     List of Record objects from the true data set
    - aRecords        List of Record objects from indexer A
    - bRecords        List of Record objects from indexer B
    - arbRecords      List of Record objects after arbitration of A and B
    - companyRecords  List of Record objects from "The Company"
    - boundingBoxes   List of RecordBoundingBoxes objects
    '''

    def __init__(self, imagePath):
        'Creates an empty ImageData'
        self.imagePath = imagePath
        self.trueRecords = []
        self.aRecords = []
        self.bRecords = []
        self.arbRecords = []
        self.companyRecords = []
        self.boundingBoxes = []

    def parseTrueXml(filepath):
        '''
        Populates the self.trueRecords list (appending)
        '''
        pass

    def parseAbarbXml(filepath):
        '''
        Populates the self.aRecords, self.bRecords, and self.arbRecords lists (appending)
        '''
        pass

    def parseCompanyXml(filepath):
        '''
        Populates the self.companyRecords and self.boundingBoxes lists (appending)
        '''
        pass

def readFiles(directory):
    '''
    Reads the files from the given directory and returns a list of ImageData objects
    '''
    pass

