import tifffile as tif
import numpy as np

def readFile(filename):
    im = tif.imread(filename)
    return im

def writeFile(arr):
    tif.imwrite('output_image.tif',arr,photometric='minisblack')

def standardDeviation(arr):
    return np.std(arr,axis=0)
    
def mean(arr):
    return np.mean(arr,axis=0)

def main():
    im = readFile('/Users/alexandrakim/Desktop/BUGS2022/example_movie.tif')
    # realStdIm = readFile('/Users/alexandrakim/Desktop/BUGS2022/Denoised_STD_20220419_ok03_abtl_h2bcamp6s_7dpf_2v3v5_2P (2).tif')
    stdIm = standardDeviation(im)
    print(stdIm)

if __name__ == "__main__":
    main()