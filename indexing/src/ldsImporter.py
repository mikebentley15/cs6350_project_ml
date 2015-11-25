'''
Utility to import LDS xml files for images to use in learning, or
anything else...
'''

from collections import namedtuple
import glob
import libxml2 as xml
import re

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
    assert len(a) > 0, 'Attribute {0} not found in node {1}'.format(attribute, node.name)
    assert len(a) < 2, 'Duplicate attributes {0} found in node {1}'.format(attribute, node.name)
    return a[0].get_content()

def getChildNode(node, name):
    a = node.xpathEval('./' + name)
    assert len(a) > 0, 'Node {0} does not have a child {1}'.format(node.name, name)
    assert len(a) < 2, 'Node {0} has more than one child {1}'.format(node.name, name)
    return a[0]

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

    def parseTrueXml(self, filepath):
        '''
        Populates the self.trueRecords list (appending)
        '''
        doc = xml.parseFile(filepath)
        headeritem = doc.xpathEval('//header-item')
        sex = []
        race = []
        married = []
        linenum = []
        r = []
        for h in headeritem:
            if getAttributeContents(h, 'name') == "PR_SEX":
                sex.append(h.get_content().strip())
            if getAttributeContents(h, 'name') == "PR_RACE_OR_COLOR":
                race.append(h.get_content().strip())
            if getAttributeContents(h, 'name') == "PR_MARITAL_STATUS":
                married.append(h.get_content().strip())
            if getAttributeContents(h, 'name') == "LINE_NBR":
                linenum.append(int(h.get_content().strip()))
        for x in xrange(len(linenum)):
            r = Record(linenum[x], sex[x], race[x], married[x])
            self.trueRecords.append(r)
        doc.freeDoc()

    def parseAbarbXml(self, filepath):
        '''
        Populates the self.aRecords, self.bRecords, and self.arbRecords lists (appending)
        '''
        doc = xml.parseFile(filepath)
        for header, records in zip(('headera', 'headerb', 'header'), (self.aRecords, self.bRecords, self.arbRecords)):
            items = doc.xpathEval('//{0}/header-item[@name="LINE_NBR"]'.format(header))
            for lineItem in items:
                recordNum = getAttributeContents(lineItem, 'record')
                similarRecords = doc.xpathEval('//{0}/header-item[@record="{1}"]'.format(header, recordNum))
                linenum = lineItem.get_content().strip()
                try:
                    linenum = int(re.search('([0-9]*)', linenum).group(0))
                except:
                    continue
                sex = ''
                race = ''
                married = ''
                for h in similarRecords:
                    if getAttributeContents(h, 'name') == "PR_SEX":
                        sex = h.get_content().strip()
                    if getAttributeContents(h, 'name') == "PR_RACE_OR_COLOR":
                        race = h.get_content().strip()
                    if getAttributeContents(h, 'name') == "PR_MARITAL_STATUS":
                        married = h.get_content().strip()
                if not (sex == '' and race == '' and married == ''):
                    records.append(Record(linenum, sex, race, married))
        doc.freeDoc()

    def parseCompanyXml(self, filepath):
        '''
        Populates the self.companyRecords and self.boundingBoxes lists (appending)
        '''
        doc = xml.parseFile(filepath)
        sex = []
        linenum = []
        race = []
        married = []
        recordNodes = doc.xpathEval('//record')
        linenumGuess = 1
        for node in recordNodes:
            try:
                lineNode = getChildNode(node, 'LINE_NBR')
                genderNode = getChildNode(node, 'PR_SEX')
                ethnicityNode = getChildNode(node, 'PR_RACE_OR_COLOR')
                maritalNode = getChildNode(node, 'MARITAL_STATUS')
            except AssertionError as ex:
                raise AssertionError('file: ' + filepath + ', ' + ex.args[0])
            linenum = lineNode.get_content().strip()
            sex = genderNode.get_content().strip()
            race = ethnicityNode.get_content().strip()
            married = maritalNode.get_content().strip()
            if linenum == '':
                linenum = linenumGuess
                #print linenum, filepath
            else:
                linenum = int(linenum)
                linenumGuess = linenum
            # In our usage of this, we are forgiving of extra company records, but not of missing company records
            # so, we err on the side of including the record even if the company "thinks" it was empty
            self.companyRecords.append(Record(linenum, sex, race, married))
            linenumGuess += 1
            # Get bounding boxes
            boxes = [] # array of sexbox, racebox, marriagebox  :)
            for subnode in (genderNode, ethnicityNode, maritalNode):
                try:
                    recNode = getChildNode(subnode, 'RecoZone')
                except AssertionError as ex:
                    raise AssertionError('file: ' + filepath + ', ' + ex.args[0])
                boxes.append(BoundingBox(
                    int(getAttributeContents(recNode, 'Top')),
                    int(getAttributeContents(recNode, 'Bottom')),
                    int(getAttributeContents(recNode, 'Left')),
                    int(getAttributeContents(recNode, 'Right'))
                    ))
            self.boundingBoxes.append(RecordBoundingBoxes(linenum, boxes[0], boxes[1], boxes[2]))
        doc.freeDoc()

def readFiles(directory):
    '''
    Reads the files from the given directory and returns a list of ImageData objects
    '''
    for name in glob.glob(directory + '/*.jpg'):
        basename = name.replace("jpg", "")
        obj = ImageData(name)
        try:
            obj.parseTrueXml(basename + 'truth.xml')
            obj.parseAbarbXml(basename + 'origABARB.xml')
            obj.parseCompanyXml(basename + 'hypBboxes.xml.filtered')
        except AssertionError as ex:
            print name, ex
        yield obj


