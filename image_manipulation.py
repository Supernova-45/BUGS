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
    def writeFile(self, arr, filename):
        tif.imwrite(filename, arr, photometric='minisblack')

    def standardDeviation(self, arr):
        return np.std(arr,axis=0,dtype='float32')
        
    def mean(self, arr):
        return np.mean(arr,axis=0,dtype='float32')

    def median(self, arr):
        return np.median(arr,axis=0,dtype='float32')

    def variance(self, arr):
        return np.var(arr,axis=0,dtype='float32')

    # need to add more