'''
Utility functions for image operations.
'''

def getSubImage(image, boundingbox):
    '''
    Returns the sub image as described by the given bounding box.
    @param image An opencv image (or rather a numpy 2D or 3D array)
    @param boundingbox A tuple in order of (top, bottom, left, right)
           You can use the BoundingBox namedtuple from ldsImporter
    @return Another opencv image only from the boundingbox
    '''
    top, bottom, left, right = boundingbox
    return image.crop((left, top, right, bottom))

def drawBox(image, boundingbox, pixelVal=0):
    '''
    Draws the bounding box on the image in black assuming the image is grayscale
    @param image An opencv image
    @param boundingbox A tuple in order of (top, bottom, left, right)
           You can use the BoundingBox namedtuple from ldsImporter
    @return None
    '''
    top, bottom, left, right = boundingbox
    image[top:bottom, left] = pixelVal
    image[top:bottom, right] = pixelVal
    image[top, left:right] = pixelVal
    image[bottom, left:right] = pixelVal
    
