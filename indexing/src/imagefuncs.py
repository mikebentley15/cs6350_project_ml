'''
Utility functions for image operations.
'''

import cv2

def convertToGray(image):
    '''
    Converts a RGB image to grayscale
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getSubImage(image, boundingbox):
    '''
    Returns the sub image as described by the given bounding box.
    @param image An opencv image (or rather a numpy 2D or 3D array)
    @param boundingbox A tuple in order of (top, bottom, left, right)
           You can use the BoundingBox namedtuple from ldsImporter
    @return Another opencv image only from the boundingbox
    '''
    top, bottom, left, right = boundingbox
    return image[top:bottom, left:right]

def mirrorPadImage(image, newSize):
    '''
    Returns a new image that has the new size by expanding the given
    image.  This assumes that the input image is smaller than or equal
    to the new size, but in case it isn't it will crop to that new
    size.

    The padding will mirror the first and last row/column equally on
    both sides.

    @param image A numpy 2D array
    @param newSize A tuple of (height, width)
    '''
    ydiff = newSize[0] - image.shape[0]
    xdiff = newSize[1] - image.shape[1]
    
    # If the image is bigger in either dim than newSize, first crop it.
    if xdiff < 0 or ydiff < 0:
        boundingBox = (
            max(0, -ydiff) / 2,                     # top
            image.shape[0] - max(0, -ydiff + 1)/2,  # bottom
            max(0, -xdiff) / 2,                     # left
            image.shape[1] - max(0, -xdiff + 1)/2,  # right
            )
        image = getSubImage(image, boundingBox)
        # Truncate them to non-negative
        xdiff = max(0, xdiff)
        ydiff = max(0, ydiff)
    
    # Pad the image
    return cv2.copyMakeBorder(image,
                              max(0, ydiff)/2,    # Added to top
                              max(0, ydiff+1)/2,  # Added to bottom
                              max(0, xdiff)/2,    # Added to left
                              max(0, xdiff+1)/2,  # Added to right
                              cv2.BORDER_REPLICATE)

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
    
