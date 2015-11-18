from collections import namedtuple
import glob
import libxml2 as xml

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

    def parseTrueXml(self,filepath):
        '''
        Populates the self.trueRecords list (appending)
        '''
        doc=xml.parseFile(filepath)
        headeritem=doc.xpathEval('//header-item')
        sex=[]
        race=[]
        married=[]
        linenum=[]
        r=[]
        for k in range(len(headeritem)):
            h=headeritem[k]
            if getAttributeContents(h,'name')=="PR_SEX":
                sex.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_RACE_OR_COLOR":
                race.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_MARITAL_STATUS":
                married.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="LINE_NBR":
                linenum.append(int(h.get_content().strip()))
        for x in range(len(linenum)):
            r=Record(linenum[x],sex[x],race[x],married[x])
            self.trueRecords.append(r)

    def parseAbarbXml(self,filepath):
        '''
        Populates the self.aRecords, self.bRecords, and self.arbRecords lists (appending)
        '''
        doc=xml.parseFile(filepath):
        aItems = doc.xpathEval('//headera/header-item')
        sex=[]
        race=[]
        married=[]
        linenum=[]
        for k in range(len(aItems)):
            h=aItems[k]
            if getAttributeContents(h,'name')=="PR_SEX":
                sex.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_RACE_OR_COLOR":
                race.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_MARITAL_STATUS":
                married.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="LINE_NBR":
                linenum.append(int(h.get_content().strip()))
        for x in range(len(linenum)):
            r=Record(linenum[x],sex[x],race[x],married[x])
            self.aRecords.append(r)
        bItems = doc.xpathEval('//headerb/header-item')
        sex=[]
        race=[]
        married=[]
        linenum=[]
        for k in range(len(bItems)):
            h=bItems[k]
            if getAttributeContents(h,'name')=="PR_SEX":
                sex.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_RACE_OR_COLOR":
                race.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_MARITAL_STATUS":
                married.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="LINE_NBR":
                linenum.append(int(h.get_content().strip()))
        for x in range(len(linenum)):
            r=Record(linenum[x],sex[x],race[x],married[x])
            self.bRecords.append(r)
        Items = doc.xpathEval('//header/header-item')
        sex=[]
        race=[]
        married=[]
        linenum=[]
        for k in range(len(Items)):
            h=Items[k]
            if getAttributeContents(h,'name')=="PR_SEX":
                sex.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_RACE_OR_COLOR":
                race.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="PR_MARITAL_STATUS":
                married.append(h.get_content().strip())
            if getAttributeContents(h,'name')=="LINE_NBR":
                linenum.append(int(h.get_content().strip()))
        for x in range(len(linenum)):
            r=Record(linenum[x],sex[x],race[x],married[x])
            self.arbRecords.append(r)
        pass

    def parseCompanyXml(self,filepath):
        '''
        Populates the self.companyRecords and self.boundingBoxes lists (appending)
        '''
        doc=xml.parseFile(filepath)
        sex=[]
        linenum=[]
        race=[]
        married=[]
        linenbr=doc.xpathEval('//record/LINE_NBR')
        gender=doc.xpathEval('//record/PR_SEX')
        ethnic=doc.xpathEval('//record/PR_RACE_OR_COLOR')
        marital=doc.xpathEval('//record/MARITAL_STATUS')
        for i in range(len(linebr)):
            h=linenbr[i]
            linenum.append(h.get_content().strip())
            h=gender[i]
            sex.append(h.get_content().strip())
            h=ethnic[i]
            race.append(h.get_content().strip())
            h=marital[i]
            married.append(h.get_content().strip())
        for x in range(len(linenum)):
            r=Record(linenum[x],sex[x],race[x],married[x])
            self.companyRecords.append(r)
        # Get the bounding Boxes
        stop=[]
        sleft=[]
        sright=[]
        sbottom=[]
        rtop=[]
        rleft=[]
        rright=[]
        rbottom=[]
        mtop=[]
        mleft=[]
        mright=[]
        mbottom=[]
        box=doc.xpathEval('//record/PR_SEX/RecoZone')
        for j in range(len(box)):
            BB=box[j]
            sbottom.append(int(getAttributeContents(BB,'Bottom')))
            stop.append(int(getAttributeContents(BB,'Top')))
            sleft.append(int(getAttributeContents(BB,'Left')))
            sright.append(int(getAttributeContents(BB,'Right')))
        box=doc.xpathEval('//record/PR_RACE_OR_COLOR/RecoZone')
        for j in range(len(box)):
            BB=box[j]
            rbottom.append(int(getAttributeContents(BB,'Bottom')))
            rtop.append(int(getAttributeContents(BB,'Top')))
            rleft.append(int(getAttributeContents(BB,'Left')))
            rright.append(int(getAttributeContents(BB,'Right')))
        box=doc.xpathEval('//record/MARITAL_STATUS/RecoZone')
        for j in range(len(box)):
            BB=box[j]
            mbottom.append(int(getAttributeContents(BB,'Bottom')))
            mtop.append(int(getAttributeContents(BB,'Top')))
            mleft.append(int(getAttributeContents(BB,'Left')))
            mright.append(int(getAttributeContents(BB,'Right')))
        #Form Bounding Boxes
        sexbox=[]
        racebox=[]
        marriedbox=[]
        for j in range(len(sbottom)):
            h=BoundingBox(stop[j],sbottom[j],sleft[j],sright[j])
            sexbox.append(h) #its only awkward if you make it
            h=BoundingBox(rtop[j],rbottom[j],rleft[j],rright[j])
            racebox.append(h) #see previous comment
            h=BoundingBox(mtop[j],mbottom[j],mleft[j],mright[j])
            marriedbox.append(h)
        for j in range(len(linenum))
            self.boundingBoxes.append(RecordBoundingBoxes(linenum[j],sexbox[j],racebox[j],marriedbox[j]))

def readFiles(directory):
    '''
    Reads the files from the given directory and returns a list of ImageData objects
    '''
    path='cs6350_project_ml/church-data/'
    namelist=glob.glob(path+'*.jpg')
    newname=[]
    Objects=[]
    for y in range(len(namelist)):
        newname.append(namelist[y].replace("jpg",""))
        for y in range(len(newname)):
            Object1=ImageData(newname[y])
            Object1.parseTrueXml(newname[y]+'truth.xml')
            Object1.parseAbarbXml(newname[y]+'origABARB.xml')
            Object1.parseCompanyXml(newname[y]+'hypBboxes.xml.filtered')
            Objects.append(Object1)
    return Objects


