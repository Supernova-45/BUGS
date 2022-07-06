import tifffile as tif
import numpy as np

def readFile(filename):
    im = tif.imread(filename)
    return im

def writeFile(arr, filename):
    tif.imwrite(filename, arr, photometric='minisblack')

def standardDeviation(arr):
    return np.std(arr,axis=0,dtype='float32')
    
def mean(arr):
    return np.mean(arr,axis=0,dtype='float32')

def median(arr):
    return np.median(arr,axis=0,dtype='float32')

def variance(arr):
    return np.var(arr,axis=0,dtype='float32')

def main():
    im = readFile('/Users/alexandrakim/Desktop/BUGS2022/example_movie.tif')
    stdIm = standardDeviation(im)
    writeFile(stdIm, 'output_image.tif')

if __name__ == "__main__":
    main()