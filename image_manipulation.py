'''
Reads and writes TIF files in addition to performing statistical manipulations on the images.
'''

import tifffile as tif
import numpy as np

class imageManipulation:

    # from file constructor
    def __init__(self, filename):
        self.im = tif.imread(filename)

    # from numpy array constructor
    def __init__(self, arr):
        self.im = arr

    def getArr(self):
        return self.im

    # statistical functions
    def writeFile(self, filename):
        tif.imwrite(filename, self.im, photometric='minisblack')

    def standardDeviation(self):
        return np.std(self.im,axis=0,dtype='float32')
        
    def mean(self):
        return np.mean(self.im,axis=0,dtype='float32')

    def median(self):
        return np.median(self.im,axis=0,dtype='float32')

    def variance(self):
        return np.var(self.im,axis=0,dtype='float32')