# from image_manipulation import imageManipulation, arrayManipulation
from cropper import Cropper, Slices

import tifffile as tif

def main():
    im = tif.imread("data/example_STD_image.tif")
    slice = Slices([0,4,4], [17,91,66], im)
    print(slice.crop(im,"data/example_STD_image.tif"))
    tif.imwrite("data/example_slice.tif", slice, photometric='minisblack')

if __name__ == "__main__":
    main()
